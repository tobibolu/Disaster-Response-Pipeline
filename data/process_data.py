"""ETL pipeline for disaster response messages.

Loads message and category data from CSV files, cleans and merges them,
then stores the result in a SQLite database.
"""

import json
import sys
from pathlib import Path

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

    required_message_columns = {'id', 'message', 'original', 'genre'}
    required_category_columns = {'id', 'categories'}
    if not required_message_columns.issubset(messages.columns):
        missing = sorted(required_message_columns - set(messages.columns))
        raise ValueError(f'Messages file is missing required columns: {missing}')
    if not required_category_columns.issubset(categories.columns):
        missing = sorted(required_category_columns - set(categories.columns))
        raise ValueError(f'Categories file is missing required columns: {missing}')

    message_only_ids = set(messages['id']) - set(categories['id'])
    category_only_ids = set(categories['id']) - set(messages['id'])
    if message_only_ids or category_only_ids:
        raise ValueError(
            'Message/category IDs do not match: '
            f'{len(message_only_ids)} message-only and '
            f'{len(category_only_ids)} category-only IDs.'
        )

    # Repeated message IDs are exact duplicate source records in this dataset.
    # Collapse them before the join so duplicate category annotations cannot
    # create a many-to-many Cartesian product.
    message_variants = messages.groupby('id')[['message', 'original', 'genre']].nunique(
        dropna=False
    )
    conflicting_message_ids = message_variants.gt(1).any(axis=1)
    if conflicting_message_ids.any():
        raise ValueError(
            f'{int(conflicting_message_ids.sum())} IDs map to conflicting message records.'
        )
    audit = {
        'source_message_rows': int(len(messages)),
        'source_category_rows': int(len(categories)),
        'source_unique_ids': int(messages['id'].nunique()),
        'repeated_message_rows': int(messages['id'].duplicated().sum()),
        'repeated_category_rows': int(categories['id'].duplicated().sum()),
        'ids_with_multiple_distinct_annotations': int(
            categories.groupby('id')['categories'].nunique().gt(1).sum()
        ),
        'duplicate_annotation_policy': 'positive_union_by_message_id',
    }
    messages = messages.drop_duplicates(subset='id', keep='first')

    merged = messages.merge(
        categories,
        how='inner',
        on='id',
        validate='one_to_many',
    )
    merged.attrs['etl_audit'] = audit
    return merged


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged DataFrame by splitting categories into binary columns.

    Args:
        df: Merged DataFrame of messages and categories.

    Returns:
        Cleaned DataFrame with 36 individual binary category columns.
    """
    if df.empty:
        raise ValueError('Cannot clean an empty merged dataset.')
    if df['categories'].isna().any():
        raise ValueError('Category strings must not be null.')

    category_tokens = df['categories'].str.split(';', expand=True)
    reference_names = [token.rsplit('-', 1)[0] for token in category_tokens.iloc[0]]
    parsed_values = pd.DataFrame(index=df.index)

    for position, category_name in enumerate(reference_names):
        parsed = category_tokens[position].str.rsplit('-', n=1, expand=True)
        if parsed.shape[1] != 2 or not parsed[0].eq(category_name).all():
            raise ValueError(
                f'Category schema mismatch at position {position}; expected {category_name!r}.'
            )
        values = pd.to_numeric(parsed[1], errors='raise')
        if not values.isin([0, 1, 2]).all():
            raise ValueError(f'Category {category_name!r} contains values outside 0, 1, or 2.')
        parsed_values[category_name] = values.clip(upper=1).astype('int8')

    parsed_values.insert(0, 'id', df['id'].to_numpy())
    # Some IDs have duplicate annotations that disagree. A positive label from
    # either annotation is retained, and the output is one row per message ID.
    category_by_id = parsed_values.groupby('id', as_index=False).max()
    message_by_id = df[['id', 'message', 'original', 'genre']].drop_duplicates('id')
    cleaned = message_by_id.merge(
        category_by_id,
        how='inner',
        on='id',
        validate='one_to_one',
    )

    # Two blank messages and four spreadsheet-error tokens contain no usable
    # language. Keeping them would create non-messages in the evaluation data.
    invalid_message = cleaned['message'].fillna('').str.strip().isin({'', '#NAME?'})
    cleaned = cleaned.loc[~invalid_message].reset_index(drop=True)
    audit = dict(df.attrs.get('etl_audit', {}))
    audit.update({
        'excluded_non_messages': int(invalid_message.sum()),
        'clean_rows': int(len(cleaned)),
        'clean_unique_ids': int(cleaned['id'].nunique()),
        'clean_duplicate_ids': int(cleaned['id'].duplicated().sum()),
        'category_columns': int(len(reference_names)),
    })
    cleaned.attrs['etl_audit'] = audit
    return cleaned


def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """Save cleaned DataFrame to a SQLite database.

    Args:
        df: Cleaned DataFrame to save.
        database_filename: Path for the SQLite database file.
    """
    Path(database_filename).resolve().parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('ETL', engine, index=False, if_exists='replace')


def save_etl_report(df: pd.DataFrame, database_filename: str) -> Path:
    """Save the data-quality decisions and row counts beside the database."""
    report_path = Path(database_filename).with_name('etl_metrics.json')
    report_path.write_text(
        json.dumps(df.attrs.get('etl_audit', {}), indent=2) + '\n',
        encoding='utf-8',
    )
    return report_path


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
        report_path = save_etl_report(df, database_filepath)

        print(f'Cleaned data saved to database!\n    REPORT: {report_path}')

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
