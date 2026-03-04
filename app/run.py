"""Flask web application for disaster response message classification.

Provides an interface to classify disaster messages into 36 categories
and displays visualizations of the training data.
"""

import os
import sys
import json

import plotly
import pandas as pd
import joblib
from flask import Flask, render_template, request
from plotly.graph_objs import Bar, Heatmap
from sqlalchemy import create_engine

# Add project root to path so we can import the shared tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import tokenize

app = Flask(__name__)

# Resolve paths relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'data', 'DisasterResponse.db')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'classifier.pkl')

engine = create_engine(f'sqlite:///{DB_PATH}')
df = pd.read_sql_table('ETL', engine)

model = joblib.load(MODEL_PATH)


@app.route('/')
@app.route('/index')
def index():
    """Render the homepage with data visualizations."""
    # Genre distribution
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Top 10 categories
    category_columns = df.columns[4:]
    top_categories = df[category_columns].sum().sort_values(ascending=False)[:10]
    top_category_names = [name.replace('_', ' ').title() for name in top_categories.index]
    top_category_counts = top_categories.values.tolist()

    # Category co-occurrence (top 10 categories)
    top_cols = df[top_categories.index]
    correlation = top_cols.corr()
    corr_labels = [name.replace('_', ' ').title() for name in correlation.columns]

    # Message length distribution by genre
    df['msg_length'] = df['message'].str.len()
    length_by_genre = df.groupby('genre')['msg_length'].mean()
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
    query = request.args.get('query', '')
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main() -> None:
    """Start the Flask web server."""
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('PORT', 3001))
    app.run(host='0.0.0.0', port=port, debug=debug)


if __name__ == '__main__':
    main()
