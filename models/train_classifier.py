"""ML pipeline for disaster response message classification.

Loads cleaned data from a SQLite database, builds a multi-output text
classification model using NLP and GridSearchCV, evaluates it, and
saves the trained model as a pickle file.
"""

import os
import sys
import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
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
    Y = df.iloc[:, 4:]

    # Drop columns with only one unique value (can't train a classifier on them)
    single_class_cols = [col for col in Y.columns if Y[col].nunique() < 2]
    if single_class_cols:
        print(f'  Dropping single-class columns: {single_class_cols}')
        Y = Y.drop(columns=single_class_cols)

    category_names = Y.columns.tolist()
    return X, Y, category_names


def build_model() -> GridSearchCV:
    """Build a text classification pipeline with GridSearchCV.

    Uses SGDClassifier with log loss (logistic regression) instead of
    RandomForest. SGD handles class imbalance better via class_weight='balanced',
    trains significantly faster on text data, and produces smaller model files.

    Returns:
        GridSearchCV model wrapping a CountVectorizer -> TF-IDF -> SGD pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_features=15000)),
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

    model = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_weighted',
                         cv=3, verbose=1)
    return model


def evaluate_model(model: GridSearchCV, X_test: pd.Series,
                   Y_test: pd.DataFrame, category_names: List[str]) -> None:
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
            target_names=['no', 'yes'], zero_division=0
        ))

    print('\n--- Overall Results ---')
    print(classification_report(
        Y_test.values, Y_pred, target_names=category_names, zero_division=0
    ))


def save_model(model: GridSearchCV, model_filepath: str) -> None:
    """Save trained model to a pickle file.

    Only saves the best estimator to minimize file size and memory usage.

    Args:
        model: Trained GridSearchCV model.
        model_filepath: Destination file path.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model.best_estimator_, f)


def main() -> None:
    """Run the ML pipeline from command line arguments."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

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
