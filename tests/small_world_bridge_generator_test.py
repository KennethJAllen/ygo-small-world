'''Tests for small_world_bridge_generator.py.'''
# pylint: disable=redefined-outer-name

import os
import numpy as np
import pandas as pd
import pytest
from ygo_small_world import small_world_bridge_generator as sw

# fixtures

@pytest.fixture
def sample_df():
    '''A sample dataframe for testing the sub_df function.'''
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'd'],
        'C': [10, 20, 30, 40, 50]
    })

@pytest.fixture
def sample_monster_names():
    '''A list sample monster names.'''
    return ['Archfiend Eccentrick', 'Ash Blossom & Joyous Spring', 'Effect Veiler', 'PSY-Framegear Gamma']

@pytest.fixture
def sample_card_df(sample_monster_names):
    '''A sample card dataframe.'''
    return pd.DataFrame({
        'id': [57624336, 14558127, 97268402, 38814750],
        'name': sample_monster_names,
        'type': ['Fiend', 'Zombie', 'Spellcaster', 'Psychic'],
        'attribute': ['LIGHT', 'FIRE', 'LIGHT', 'LIGHT'],
        'level': [3.0, 3.0, 1.0, 2.0],
        'atk': [800.0, 0.0, 0.0, 1000.0],
        'def': [1000.0, 1800.0, 0.0, 0.0]
    })

@pytest.fixture
def ydk_file_path():
    '''The path of the test ydk file.'''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ydk_file = 'test_deck.ydk'
    ydk_file_path = os.path.join(current_dir, ydk_file)
    return ydk_file_path

@pytest.fixture
def sample_adjacency_matrix():
    '''A sample adjacency matrix.'''
    return np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])

@pytest.fixture
def sample_adjacency_matrix_squared():
    '''A sample adjacency matrix squared.'''
    return np.array([[3, 1, 1, 0], [1, 2, 1, 1], [1, 1, 2, 1], [0, 1, 1, 1]])

@pytest.fixture
def sample_labeled_adjacency_matrix(sample_adjacency_matrix, sample_monster_names):
    '''A sample labeled adjacency matrix.'''
    return pd.DataFrame(sample_adjacency_matrix, index=sample_monster_names, columns=sample_monster_names)

@pytest.fixture
def sample_labeled_adjacency_matrix_squared(sample_adjacency_matrix_squared, sample_monster_names):
    '''A sample labeled adjacency matrix squared.'''
    return pd.DataFrame(sample_adjacency_matrix_squared, index=sample_monster_names, columns=sample_monster_names)

# Test sub_df

def test_valid_subset_ints(sample_df):
    '''Test subset with integer values in column.'''
    result = sw.sub_df(sample_df, column_values=[2, 4], column_name='A').reset_index(drop=True)
    expected = pd.DataFrame({
        'A': [2, 4],
        'B': ['b', 'd'],
        'C': [20, 40]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_valid_subset_strs(sample_df):
    '''Test subset with string values in column.'''
    result = sw.sub_df(sample_df, column_values=['b', 'd'], column_name='B').reset_index(drop=True)
    expected = pd.DataFrame({
        'A': [2, 4, 5],
        'B': ['b', 'd', 'd'],
        'C': [20, 40, 50]
    })
    pd.testing.assert_frame_equal(result, expected)

def test_empty_subset(sample_df):
    '''Test empty subset.'''
    with pytest.raises(ValueError, match=r"Not all values are in column."):
        sw.sub_df(sample_df, column_values=[6, 7], column_name='A')

def test_invalid_column(sample_df):
    '''Test invalid column name.'''
    with pytest.raises(ValueError, match=r"'D' is not a valid column in the DataFrame."):
        sw.sub_df(sample_df, column_values=[1, 2], column_name='D')

def test_column_values_not_list(sample_df):
    '''Test value not in column.'''
    with pytest.raises(TypeError):
        sw.sub_df(sample_df, column_values="a", column_name='B')

# test load_main_monsters

def test_main_monsters_notnull():
    '''Test there are no nulls.'''
    main_monsters = sw.load_main_monsters()
    assert main_monsters.notnull().values.all()

# test monster_names_to_df

def test_monster_names_to_df(sample_monster_names, sample_card_df):
    '''Test getting df from sample monster names'''
    result = sw.monster_names_to_df(sample_monster_names).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, sample_card_df)

# test ydk_to_card_ids

def test_ydk_to_card_ids(ydk_file_path):
    '''Test getting card ids from test ydk file'''
    result = sw.ydk_to_card_ids(ydk_file_path)
    print(ydk_file_path)
    expected = [14558127, 97268402, 14558127, 38814750, 57624336, 14558127]
    assert result == expected

# test ydk_to_monster_names

def test_ydk_to_monster_names(sample_monster_names, ydk_file_path):
    '''Test getting card ids from test ydk file'''
    result = sw.ydk_to_monster_names(ydk_file_path)
    expected = sample_monster_names
    assert result == expected

# test df_to_adjacency_matrix

def test_df_to_adjacency_matrix(sample_card_df, sample_adjacency_matrix):
    '''Test generating adjacency matrix from dataframe.'''
    result = sw.df_to_adjacency_matrix(sample_card_df)
    np.testing.assert_array_equal(result, sample_adjacency_matrix)

def test_df_to_adjacency_matrix_squared(sample_card_df, sample_adjacency_matrix_squared):
    '''Test generating squared adjacency matrix from dataframe.'''
    result = sw.df_to_adjacency_matrix(sample_card_df, squared=True)
    np.testing.assert_array_equal(result, sample_adjacency_matrix_squared)

# test names_to_adjacency_matrix

def test_names_to_adjacency_matrix(sample_monster_names, sample_adjacency_matrix):
    '''Test generating adjacency matrix from list of monster names.'''
    result = sw.names_to_adjacency_matrix(sample_monster_names)
    np.testing.assert_array_equal(result, sample_adjacency_matrix)

def test_names_to_adjacency_matrix_squared(sample_monster_names, sample_adjacency_matrix_squared):
    '''Test generating squared adjacency matrix from list of monster names.'''
    result = sw.names_to_adjacency_matrix(sample_monster_names, squared=True)
    np.testing.assert_array_equal(result, sample_adjacency_matrix_squared)

# test names_to_labeled_adjacency_matrix

def test_names_to_labeled_adjacency_matrix(sample_monster_names, sample_labeled_adjacency_matrix):
    '''Test generating adjacency matrix from list of monster names.'''
    result = sw.names_to_labeled_adjacency_matrix(sample_monster_names)
    np.testing.assert_array_equal(result, sample_labeled_adjacency_matrix)

def test_names_to_labeled_adjacency_matrix_squared(sample_monster_names, sample_labeled_adjacency_matrix_squared):
    '''Test generating squared adjacency matrix from list of monster names.'''
    result = sw.names_to_labeled_adjacency_matrix(sample_monster_names, squared=True)
    np.testing.assert_array_equal(result, sample_labeled_adjacency_matrix_squared)

# test ydk_to_labeled_adjacency_matrix

def test_ydk_to_labeled_adjacency_matrix(ydk_file_path, sample_labeled_adjacency_matrix):
    '''Test generating adjacency matrix from list of monster names.'''
    result = sw.ydk_to_labeled_adjacency_matrix(ydk_file_path)
    np.testing.assert_array_equal(result, sample_labeled_adjacency_matrix)

def test_ydk_to_labeled_adjacency_matrix_squared(ydk_file_path, sample_labeled_adjacency_matrix_squared):
    '''Test generating squared adjacency matrix from list of monster names.'''
    result = sw.ydk_to_labeled_adjacency_matrix(ydk_file_path, squared=True)
    np.testing.assert_array_equal(result, sample_labeled_adjacency_matrix_squared)

# test find_best_bridges.
# The best bridges could change with newly released cards, so certian properties are tested

def test_length_find_best_bridges(sample_monster_names):
    '''Checks that length of best bridges is sufficiently large.'''
    result = sw.find_best_bridges(sample_monster_names)
    assert len(result) > 4000

def test_number_of_connections(sample_monster_names):
    '''
    Checks that the most number of connections is equal to the number of monsters.
    This test only works if there is a small world bridge that connects to all monsters in sample_monster_names.
    '''
    result = sw.find_best_bridges(sample_monster_names)
    num_monsters = len(sample_monster_names)
    assert result['number_of_connections'].max() == num_monsters

def test_bridge_score(sample_monster_names):
    '''
    Checks that the best bridge score is 1.0.
    This test only works if there is a small world bridge that connects to all monsters in sample_monster_names.
    '''
    result = sw.find_best_bridges(sample_monster_names)
    assert result['bridge_score'].max() == 1.0

def test_all_required_target_names(sample_monster_names):
    '''
    Tests that smallest number of connections with all required targets is equal to the number of monsters.
    This test only works if there is a small world bridge that connects to all monsters in sample_monster_names.
    '''
    result = sw.find_best_bridges(sample_monster_names, required_target_names=sample_monster_names)
    num_required_targets = len(sample_monster_names)
    assert result['number_of_connections'].min() == num_required_targets

def test_some_required_target_names(sample_monster_names):
    '''Tests that smallest number of connections with all required targets is equal to num_in_sublist.'''
    num_in_sublist = 2
    required_target_names = sample_monster_names[:num_in_sublist]
    result = sw.find_best_bridges(sample_monster_names, required_target_names=required_target_names)
    assert result['number_of_connections'].min() == num_in_sublist
