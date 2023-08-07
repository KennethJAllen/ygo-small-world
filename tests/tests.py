import pandas as pd
import pytest
from ygo_small_world import small_world_functions as sw

def test_sub_df():
    # Creating a sample DataFrame
    data = {'A': [1, 2, 3, 2, 1], 'B': [5, 6, 7, 8, 9]}
    df = pd.DataFrame(data)
    
    # Testing sub_df function with column_name='A' and column_values=[1, 2]
    result = sw.sub_df(df, [1, 2], 'A')
    expected_result = pd.DataFrame({'A': [1, 2, 2, 1], 'B': [5, 6, 8, 9]})
    pd.testing.assert_frame_equal(result, expected_result)

    # Test with column_values that don't match any values in the specified column
    result = sw.sub_df(df, [4, 5], 'A')
    assert result.empty

    # Test with invalid column_name
    with pytest.raises(KeyError):
        sw.sub_df(df, [1, 2], 'C')

test_sub_df()