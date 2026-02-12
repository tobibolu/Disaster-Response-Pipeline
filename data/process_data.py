"""ETL pipeline for disaster response messages.

Loads message and category data from CSV files, cleans and merges them,
then stores the result in a SQLite database.
"""

import sys
from typing import Tuple

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """Load and merge messages and categories datasets.

    Args:
        messages_filepath: Path to the messages CSV file.
        categories_filepath: Path to the categories CSV file.

    Returns:
        Merged DataFrame of messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, how='outer', on='id')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged DataFrame by splitting categories into binary columns.

    Args:
        df: Merged DataFrame of messages and categories.

    Returns:
        Cleaned DataFrame with 36 individual binary category columns.
    """
    categories = df.categories.str.split(';', expand=True)

    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)

    # Clip values to binary (the 'related' column contains some 2s)
    categories = categories.clip(upper=1)

    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    return df


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """Save cleaned DataFrame to a SQLite database.

    Args:
        df: Cleaned DataFrame to save.
        database_filename: Path for the SQLite database file.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('ETL', engine, index=False, if_exists='replace')


def main() -> None:
    """Run the ETL pipeline from command line arguments."""
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}'
              f'\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print(
            'Please provide the filepaths of the messages and categories '
            'datasets as the first and second argument respectively, as '
            'well as the filepath of the database to save the cleaned data '
            'to as the third argument.\n\n'
            'Example: python data/process_data.py '
            'data/disaster_messages.csv data/disaster_categories.csv '
            'data/DisasterResponse.db'
        )


if __name__ == '__main__':
    main()
