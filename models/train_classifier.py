"""ML pipeline for disaster response message classification.

Loads cleaned data from a SQLite database, builds a multi-output text
classification model using NLP and GridSearchCV, evaluates it, and
saves the trained model as a pickle file.
"""

import json
import os
import platform
import sys
from pathlib import Path
from typing import Tuple, List

import joblib
import pandas as pd
import sklearn
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Add project root to path so we can import the shared tokenizer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import tokenize


def load_data(database_filepath: str) -> Tuple[pd.Series, pd.DataFrame, List[str]]:
    """Load data from SQLite database.

    Drops any category columns that contain only a single class (e.g.
    'child_alone' which has zero positive examples), since classifiers
    require at least two classes to train on.

    Args:
        database_filepath: Path to the SQLite database file.

    Returns:
        Tuple of (messages, category labels, category column names).
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('ETL', engine)
    X = df['message']
    non_category_columns = {'id', 'message', 'original', 'genre'}
    category_names = [col for col in df.columns if col not in non_category_columns]
    if not category_names:
        raise ValueError('No category columns were found in the ETL table.')
    Y = df[category_names]

    # Drop columns with only one unique value (can't train a classifier on them)
    single_class_cols = [col for col in Y.columns if Y[col].nunique() < 2]
    if single_class_cols:
        print(f'  Dropping single-class columns: {single_class_cols}')
        Y = Y.drop(columns=single_class_cols)

    return X, Y, Y.columns.tolist()


def build_model() -> GridSearchCV:
    """Build a text classification pipeline with GridSearchCV.

    Uses SGDClassifier with log loss (logistic regression) instead of
    RandomForest. SGD handles class imbalance better via class_weight='balanced',
    trains significantly faster on text data, and produces smaller model files.

    Returns:
        GridSearchCV model wrapping a CountVectorizer -> TF-IDF -> SGD pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            max_features=15000,
        )),
        ('tfidf', TfidfTransformer(sublinear_tf=True)),
        ('clf', MultiOutputClassifier(
            SGDClassifier(loss='log_loss', class_weight='balanced',
                          random_state=42, n_jobs=1)
        ))
    ])

    parameters = {
        'clf__estimator__alpha': [1e-4, 1e-3],
        'clf__estimator__max_iter': [1000],
        'tfidf__use_idf': [True],
    }

    cross_validation = MultilabelStratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )
    model = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring='f1_weighted',
        cv=cross_validation,
        verbose=1,
    )
    return model


def split_data(
    X: pd.Series,
    Y: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Create a deterministic split that preserves rare multilabel prevalence.

    Ordinary random splitting can leave rare emergency categories out of the
    holdout. Iterative stratification balances all label columns jointly while
    still keeping every message in exactly one partition.
    """
    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_index, test_index = next(splitter.split(X, Y))
    return (
        X.iloc[train_index],
        X.iloc[test_index],
        Y.iloc[train_index],
        Y.iloc[test_index],
    )


def evaluate_model(model: GridSearchCV, X_test: pd.Series,
                   Y_test: pd.DataFrame, category_names: List[str]) -> dict:
    """Print classification report for each category and overall.

    Args:
        model: Trained model.
        X_test: Test messages.
        Y_test: True labels for test messages.
        category_names: Names of the 36 category columns.
    """
    Y_pred = model.predict(X_test)

    print('\n--- Per-Category Results ---')
    for i, category in enumerate(category_names):
        print(f'\n{category}:')
        print(classification_report(
            Y_test.values[:, i], Y_pred[:, i],
            labels=[0, 1], target_names=['no', 'yes'], zero_division=0
        ))

    print('\n--- Overall Results ---')
    print(classification_report(
        Y_test.values, Y_pred, target_names=category_names, zero_division=0
    ))

    precision, recall, f1, support = precision_recall_fscore_support(
        Y_test.values,
        Y_pred,
        average=None,
        zero_division=0,
    )
    per_category = {
        category: {
            'precision': float(precision[index]),
            'recall': float(recall[index]),
            'f1': float(f1[index]),
            'positive_support': int(Y_test.iloc[:, index].sum()),
            'predicted_positive': int(Y_pred[:, index].sum()),
        }
        for index, category in enumerate(category_names)
    }
    aggregate_metrics = {}
    for average in ('micro', 'macro', 'weighted', 'samples'):
        avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
            Y_test.values,
            Y_pred,
            average=average,
            zero_division=0,
        )
        aggregate_metrics[average] = {
            'precision': float(avg_precision),
            'recall': float(avg_recall),
            'f1': float(avg_f1),
        }
    return {
        'holdout_rows': int(len(Y_test)),
        'exact_match_accuracy': float(accuracy_score(Y_test.values, Y_pred)),
        'hamming_loss': float(hamming_loss(Y_test.values, Y_pred)),
        'micro_f1': float(f1_score(Y_test.values, Y_pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(Y_test.values, Y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(
            f1_score(Y_test.values, Y_pred, average='weighted', zero_division=0)
        ),
        'aggregate_metrics': aggregate_metrics,
        'per_category': per_category,
    }


def save_model(model: GridSearchCV, model_filepath: str,
               category_names: List[str]) -> None:
    """Save trained model to a pickle file.

    Only saves the best estimator to minimize file size and memory usage.

    Args:
        model: Trained GridSearchCV model.
        model_filepath: Destination file path.
    """
    destination = Path(model_filepath)
    destination.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        'model': model.best_estimator_,
        'category_names': category_names,
        'metadata': {
            'python_version': platform.python_version(),
            'scikit_learn_version': sklearn.__version__,
            'best_cv_weighted_f1': float(model.best_score_),
            'best_params': model.best_params_,
        },
    }
    joblib.dump(artifact, destination)


def save_metrics(metrics: dict, model_filepath: str) -> Path:
    """Write machine-readable evaluation evidence beside the model artifact."""
    metrics_path = Path(model_filepath).with_name('model_metrics.json')
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    return metrics_path


def main() -> None:
    """Run the ML pipeline from command line arguments."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = split_data(X, Y)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        metrics = evaluate_model(model, X_test, Y_test, category_names)
        metrics.update({
            'source_rows': int(len(X)),
            'training_rows': int(len(X_train)),
            'category_count': int(len(category_names)),
            'single_class_categories_excluded': ['child_alone'],
            'best_cv_weighted_f1': float(model.best_score_),
            'best_params': model.best_params_,
            'python_version': platform.python_version(),
            'scikit_learn_version': sklearn.__version__,
        })

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath, category_names)
        metrics_path = save_metrics(metrics, model_filepath)

        print(f'Trained model saved!\n    METRICS: {metrics_path}')

    else:
        print(
            'Please provide the filepath of the disaster messages database '
            'as the first argument and the filepath of the pickle file to '
            'save the model to as the second argument.\n\n'
            'Example: python models/train_classifier.py '
            'data/DisasterResponse.db models/classifier.pkl'
        )


if __name__ == '__main__':
    main()
