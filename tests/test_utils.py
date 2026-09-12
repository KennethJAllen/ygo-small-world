"""Tests for utils.py."""

import pytest
import pandas as pd
from ygo_small_world import utils

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


def test_ydk_sections_comments_and_bom(tmp_path):
    path = tmp_path / 'sections.ydk'
    path.write_text('\ufeff#created by test\n99\n#main\n1\n#comment\n\n2\n1\n#extra\n3\n!side\n4\n', encoding='utf-8')
    assert utils.ydk_to_card_ids(path) == [1, 2, 1]


@pytest.mark.parametrize('text', ['', '1\n2\n', '#extra\n1\n!side\n2\n', '#main\n#comment\n#extra\n1\n'])
def test_ydk_requires_nonempty_main(tmp_path, text):
    path = tmp_path / 'empty.ydk'
    path.write_text(text)
    with pytest.raises(ValueError, match='no main-deck card IDs'):
        utils.ydk_to_card_ids(path)
