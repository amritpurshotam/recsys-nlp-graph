"""
Parses item to item relationships in 'related' field and explodes it such that each relationship is a single row.
"""
import argparse
import ast

import numpy as np
import pandas as pd

from src.utils.logger import logger


def explode_on(df: pd.DataFrame, relationship: str) -> pd.DataFrame:
    exploded_df = df[['asin', relationship]].explode(relationship)
    exploded_df = exploded_df.rename({relationship: "related"}, axis=1)
    exploded_df['relationship'] = relationship
    return exploded_df


def get_node_relationship(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe of products and their relationships (e.g., bought together, also bought, also viewed)

    Args:
        df:

    Returns:

    """
    # Keep only rows with related data
    df = df[~df['also_buy'].isnull()].copy()
    df = df[~df['also_view'].isnull()].copy()
    logger.info('DF shape after dropping empty also_buy and also_view: {}'.format(df.shape))

    df = df[~df['title'].isnull()].copy()
    logger.info('DF shape after dropping empty title: {}'.format(df.shape))
    df = df[['asin', 'also_buy', 'also_view']].copy()

    # Evaluate str columns into lists
    df['also_buy'] = df['also_buy'].apply(ast.literal_eval)
    df['also_view'] = df['also_view'].apply(ast.literal_eval)
    logger.info('Completed eval on "also_buy" and "also_view" string')

    # Exclude products where also_buy relationship less than 2 at least one view
    df = df[df['also_buy'].str.len() >= 2].copy()
    df = df[df['also_view'].str.len() >= 1].copy()
    logger.info('DF shape after dropping products with <2 edges: {}'.format(df.shape))

    # Explode columns
    also_buy_df = explode_on(df, relationship='also_buy')
    also_view_df = explode_on(df, relationship='also_view')

    # Concatenate df
    combined_df = pd.concat([also_buy_df, also_view_df], axis=0)
    logger.info('Distribution of relationships: \n{}'.format(combined_df['relationship'].value_counts()))

    return combined_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preparing node relationships')
    parser.add_argument('read_path', type=str, help='Path to input csv')
    parser.add_argument('write_path', type=str, help='Path to output csv (of nodes relationships)')
    args = parser.parse_args()

    df = pd.read_csv(
        args.read_path,
        on_bad_lines='warn',
        usecols=['asin', 'also_buy', 'also_view', 'title'],
        dtype={
            'asin': 'str',
            'title': 'str'
        })
    logger.info('DF shape: {}'.format(df.shape))

    exploded_df = get_node_relationship(df)

    exploded_df.to_csv(args.write_path, index=False)
    logger.info('Csv saved to {}'.format(args.write_path))
