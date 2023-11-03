import pandas as pd
import pytest
from ygo_small_world import small_world_functions as sw

# Sample data for tests
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [10, 20, 30, 40, 50]
    })

def test_valid_subset_ints(sample_df):
    result = sw.sub_df(sample_df, column_values=[2, 4], column_name='A')
    expected = pd.DataFrame({
        'A': [2, 4],
        'B': ['b', 'd'],
        'C': [20, 40]
    })
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

def test_valid_subset_strs(sample_df):
    result = sw.sub_df(sample_df, column_values=['b', 'd'], column_name='B')
    expected = pd.DataFrame({
        'A': [2, 4],
        'B': ['b', 'd'],
        'C': [20, 40]
    })
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))

def test_empty_subset(sample_df):
    result = sw.sub_df(sample_df, column_values=[6, 7], column_name='A')
    assert result.empty

def test_invalid_column(sample_df):
    with pytest.raises(ValueError, match=r"'D' is not a valid column in the DataFrame."):
        sw.sub_df(sample_df, column_values=[1, 2], column_name='D')

def test_column_values_not_list(sample_df):
    with pytest.raises(TypeError):
        sw.sub_df(sample_df, column_values="a", column_name='B')