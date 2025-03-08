"""Tests for utils.py."""
from pathlib import Path
import pytest
import pandas as pd
from ygo_small_world import utils

@pytest.fixture(name='sample_df')
def fixture_sample_df():
    '''A sample dataframe for testing the sub_df function.'''
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'd'],
        'C': [10, 20, 30, 40, 50]
    })

@pytest.fixture(name='ydk_file_path')
def fixture_ydk_file_path():
    """The path of the test ydk file."""
    ydk_file_path = Path(__file__).parent / 'test_data' / 'test_deck.ydk'
    return ydk_file_path

# Test sub_df

def test_valid_subset_ints(sample_df):
    '''Test subset with integer values in column.'''
    result = utils.sub_df(sample_df, column_values=[2, 4], column_name='A').reset_index(drop=True)
    expected = pd.DataFrame({
        'A': [2, 4],
        'B': ['b', 'd'],
        'C': [20, 40]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_valid_subset_strs(sample_df):
    '''Test subset with string values in column.'''
    result = utils.sub_df(sample_df, column_values=['b', 'd'], column_name='B').reset_index(drop=True)
    expected = pd.DataFrame({
        'A': [2, 4, 5],
        'B': ['b', 'd', 'd'],
        'C': [20, 40, 50]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_invalid_column(sample_df):
    '''Test invalid column name.'''
    with pytest.raises(ValueError, match=r"'D' is not a valid column in the DataFrame."):
        utils.sub_df(sample_df, column_values=[1, 2], column_name='D')

def test_column_values_not_list(sample_df):
    '''Test value not in column.'''
    with pytest.raises(TypeError):
        utils.sub_df(sample_df, column_values="a", column_name='B')

# test ydk_to_card_ids

def test_ydk_to_card_ids(ydk_file_path):
    '''Test getting card ids from test ydk file'''
    result = utils.ydk_to_card_ids(ydk_file_path)
    print(ydk_file_path)
    expected = [14558127, 97268402, 14558127, 38814750, 57624336, 14558127]
    assert result == expected
