import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - Filepath for the messages dataset
    categories_filepath - Filepath for the categories dataset
    OUTPUT:
    df - Merged dataframe of messages and categories datasets
    '''
    messages = pd.read_csv(messages_filepath)
    messages.head()
    categories = pd.read_csv(categories_filepath)
    categories.head()

    #Merge Datasets into one dataframe
    df = messages.merge(categories, how = 'outer', on = 'id')
    return df

def clean_data(df):
    '''
    INPUT:
    df - Dataframe of messages and categories Datasets
    OUTPUT:
    df - cleaned dataframe with new category columns
    '''
    #Create Dataframe of 36 individual category columns
    categories = df.categories.str.split(';', expand = True)

    #Extract list of new column names for categories
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x : x[:-2])
    categories.columns = category_colnames

    #Converting Category values to 0's and 1's
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    #Replace categories column in df with new categories
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories],axis=1)

    #Remove Duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    INPUT:
    df - Cleaned dataframe
    database_filename - Name of database file to save to
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('ETL', engine, index=False)
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
