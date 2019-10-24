import sys
import pickle
import re
import sqlite3
import random
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - saved database from process_data.py
    OUTPUT:
    X - Dataframe of messages column
    Y - Dataframe of all different categories
    category_names - Column names of Y Dataframe
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('ETL', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    INPUT:
    text - sentence to be analysed
    OUTPUT:
    cleaned_words - Normalized text
    '''

    #Remove punctuatuations and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())

    #Split text into words
    words = word_tokenize(text)

    #Remove Stop Words
    words = [w for w in words if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()

    cleaned_words = []
    for word in words:
        #lemmatize words
        clean_word = lemmatizer.lemmatize(word).strip()
        cleaned_words.append(clean_word)
    return cleaned_words


def build_model():
    '''
    INPUT:
    None
    OUTPUT:
    model - machine learning pipeline for text
    '''

    #Transform Data with CountVectorizer and TfidfTransformer
    #Fit Classifier
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #Modify Pipeline with GridSearch to choose optimal parameters
    parameters = {'clf__estimator__max_depth': [2, None],
                  'clf__estimator__n_estimators': [10, 50]
                 }
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - machine learning model
    X_test - Test data set
    Y_test - Set of labels to data in X_test
    category_names - Column names for Y dataframe 
    '''
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]),         target_names=category_names))
    pass


def save_model(model, model_filepath):
    '''
    INPUT:
    model - machine learning model
    model_filepath - filepath to save model to
    '''
    pickle.dump(model, open(model_filepath, "wb"))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        from workspace_utils import active_session
        with active_session():
            model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
