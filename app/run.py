"""Flask web application for disaster response message classification.

Provides an interface to classify disaster messages into 36 categories
and displays visualizations of the training data.
"""

import json
import os
import sys
from pathlib import Path

import plotly
import pandas as pd
import joblib
from flask import Flask, jsonify, render_template, request
from plotly.graph_objs import Bar, Heatmap
from sqlalchemy import create_engine

# Add project root to path so we can import the shared tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import tokenize

app = Flask(__name__)

# Resolve paths relative to this script's location, while allowing temporary
# artifacts to be injected during smoke tests and deployments.
PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get(
    'DISASTER_DB_PATH',
    PROJECT_DIR / 'data' / 'DisasterResponse.db',
))
MODEL_PATH = Path(os.environ.get(
    'DISASTER_MODEL_PATH',
    PROJECT_DIR / 'models' / 'classifier.pkl',
))

if not DB_PATH.exists():
    raise FileNotFoundError(
        f'Database not found at {DB_PATH}. Run data/process_data.py before starting the app.'
    )
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f'Model not found at {MODEL_PATH}. Run models/train_classifier.py before starting the app.'
    )

engine = create_engine(f'sqlite:///{DB_PATH.resolve()}')
df = pd.read_sql_table('ETL', engine)

try:
    model_artifact = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(
        'The classifier could not be loaded. Recreate the Python 3.11 environment '
        'from requirements.txt and retrain the model; scikit-learn pickles are '
        'version-sensitive.'
    ) from exc

non_category_columns = {'id', 'message', 'original', 'genre'}
all_category_columns = [col for col in df.columns if col not in non_category_columns]
if isinstance(model_artifact, dict) and 'model' in model_artifact:
    model = model_artifact['model']
    trained_categories = model_artifact['category_names']
    model_metadata = model_artifact.get('metadata', {})
else:
    # Backward compatibility for the earlier pipeline-only pickle. New training
    # runs persist category order explicitly to prevent silent label remapping.
    model = model_artifact
    trained_categories = [
        col for col in all_category_columns if df[col].nunique() >= 2
    ]
    model_metadata = {'artifact_format': 'legacy_pipeline_only'}

unknown_categories = sorted(set(trained_categories) - set(all_category_columns))
if unknown_categories:
    raise ValueError(
        f'Model categories are missing from the ETL table: {unknown_categories}'
    )
trained_category_indexes = {
    category: index for index, category in enumerate(trained_categories)
}


@app.route('/')
@app.route('/index')
def index():
    """Render the homepage with data visualizations."""
    # Genre distribution
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Top 10 categories
    top_categories = df[all_category_columns].sum().sort_values(ascending=False)[:10]
    top_category_names = [name.replace('_', ' ').title() for name in top_categories.index]
    top_category_counts = top_categories.values.tolist()

    # Category co-occurrence (top 10 categories)
    top_cols = df[top_categories.index]
    correlation = top_cols.corr()
    corr_labels = [name.replace('_', ' ').title() for name in correlation.columns]

    # Message length distribution by genre
    message_lengths = df.assign(msg_length=df['message'].str.len())
    length_by_genre = message_lengths.groupby('genre')['msg_length'].mean()
    length_genre_names = list(length_by_genre.index)
    length_genre_values = length_by_genre.values.tolist()

    graphs = [
        {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Genre'},
                'template': 'plotly_white'
            }
        },
        {
            'data': [Bar(
                x=top_category_names,
                y=top_category_counts,
                marker={'color': 'rgb(55, 128, 191)'}
            )],
            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Category', 'tickangle': -30},
                'template': 'plotly_white'
            }
        },
        {
            'data': [Heatmap(
                z=correlation.values.tolist(),
                x=corr_labels,
                y=corr_labels,
                colorscale='RdBu',
                zmin=-1, zmax=1
            )],
            'layout': {
                'title': 'Category Correlation Heatmap (Top 10)',
                'xaxis': {'tickangle': -30},
                'template': 'plotly_white',
                'height': 500
            }
        },
        {
            'data': [Bar(
                x=length_genre_names,
                y=length_genre_values,
                marker={'color': 'rgb(44, 160, 101)'}
            )],
            'layout': {
                'title': 'Average Message Length by Genre',
                'yaxis': {'title': 'Average Characters'},
                'xaxis': {'title': 'Genre'},
                'template': 'plotly_white'
            }
        }
    ]

    ids = [f'graph-{i}' for i in range(len(graphs))]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """Handle user query and display classification results."""
    query = request.args.get('query', '').strip()
    if not query:
        return render_template(
            'go.html',
            query=query,
            classification_result={},
            message_error='Enter a disaster-response message before classifying.',
        ), 400

    classification_labels = model.predict([query])[0]
    if len(classification_labels) != len(trained_categories):
        raise RuntimeError(
            'Model output width does not match its saved category metadata.'
        )

    # Map predictions to the trained categories, and default dropped
    # categories (e.g. child_alone) to 0
    classification_results = {}
    for col in all_category_columns:
        if col in trained_category_indexes:
            idx = trained_category_indexes[col]
            classification_results[col] = int(classification_labels[idx])
        else:
            classification_results[col] = 0

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


@app.route('/health')
def health():
    """Expose a lightweight readiness check for local and container runs."""
    return jsonify({
        'status': 'ok',
        'rows': int(len(df)),
        'displayed_categories': len(all_category_columns),
        'trained_categories': len(trained_categories),
        'model_metadata': model_metadata,
    })


def main() -> None:
    """Start the Flask web server."""
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('PORT', 3001))
    app.run(host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    main()
