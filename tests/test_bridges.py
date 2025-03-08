'''Tests for small_world_bridge_generator.py.'''

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# fixtures

@pytest.fixture(name='sample_monster_names')
def fixture_sample_monster_names():
    '''A list sample monster names.'''
    return ['Archfiend Eccentrick', 'Ash Blossom & Joyous Spring', 'Effect Veiler', 'PSY-Framegear Gamma']

@pytest.fixture(name='sample_card_df')
def fixture_sample_card_df(sample_monster_names):
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

@pytest.fixture(name='ydk_file_path')
def fixture_ydk_file_path():
    """The path of the test ydk file."""
    ydk_file_path = Path(__file__).parent / 'test_data' / 'test_deck.ydk'
    return ydk_file_path

@pytest.fixture(name='sample_adjacency_matrix')
def fixture_sample_adjacency_matrix():
    '''A sample adjacency matrix.'''
    return np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])

@pytest.fixture(name='sample_adjacency_matrix_squared')
def fixture_sample_adjacency_matrix_squared():
    '''A sample adjacency matrix squared.'''
    return np.array([[3, 1, 1, 0], [1, 2, 1, 1], [1, 1, 2, 1], [0, 1, 1, 1]])

@pytest.fixture(name='sample_labeled_adjacency_matrix')
def fixture_sample_labeled_adjacency_matrix(sample_adjacency_matrix, sample_monster_names):
    '''A sample labeled adjacency matrix.'''
    return pd.DataFrame(sample_adjacency_matrix, index=sample_monster_names, columns=sample_monster_names)

@pytest.fixture(name='sample_labeled_adjacency_matrix_squared')
def fixture_sample_labeled_adjacency_matrix_squared(sample_adjacency_matrix_squared, sample_monster_names):
    '''A sample labeled adjacency matrix squared.'''
    return pd.DataFrame(sample_adjacency_matrix_squared, index=sample_monster_names, columns=sample_monster_names)

# test load_main_monsters

def test_main_monsters_notnull(all_cards):
    '''Test there are no nulls.'''
    assert len(all_cards.get_df()) > 0

# test monster_names_to_df

def test_monster_names_to_df(sample_monster_names, sample_card_df):
    '''Test getting df from sample monster names'''
    result = bridges.monster_names_to_df(sample_monster_names).reset_index(drop=True)
    small_world_columns = ['id', 'name', 'type', 'attribute', 'level', 'atk', 'def']
    pd.testing.assert_frame_equal(result[small_world_columns], sample_card_df)

# test ydk_to_monster_names

def test_ydk_to_monster_names(sample_monster_names, ydk_file_path):
    '''Test getting card ids from test ydk file'''
    result = bridges.ydk_to_monster_names(ydk_file_path)
    expected = sample_monster_names
    assert result == expected

# test df_to_adjacency_matrix

def test_df_to_adjacency_matrix(sample_card_df, sample_adjacency_matrix):
    '''Test generating adjacency matrix from dataframe.'''
    result = bridges.df_to_adjacency_matrix(sample_card_df)
    np.testing.assert_array_equal(result, sample_adjacency_matrix)

def test_df_to_adjacency_matrix_squared(sample_card_df, sample_adjacency_matrix_squared):
    '''Test generating squared adjacency matrix from dataframe.'''
    result = bridges.df_to_adjacency_matrix(sample_card_df, squared=True)
    np.testing.assert_array_equal(result, sample_adjacency_matrix_squared)

# test names_to_adjacency_matrix

def test_names_to_adjacency_matrix(sample_monster_names, sample_adjacency_matrix):
    '''Test generating adjacency matrix from list of monster names.'''
    result = bridges.card_ids_to_adjacency_matrix(sample_monster_names)
    np.testing.assert_array_equal(result, sample_adjacency_matrix)

def test_names_to_adjacency_matrix_squared(sample_monster_names, sample_adjacency_matrix_squared):
    '''Test generating squared adjacency matrix from list of monster names.'''
    result = bridges.card_ids_to_adjacency_matrix(sample_monster_names, squared=True)
    np.testing.assert_array_equal(result, sample_adjacency_matrix_squared)

# test names_to_labeled_adjacency_matrix

def test_names_to_labeled_adjacency_matrix(sample_monster_names, sample_labeled_adjacency_matrix):
    '''Test generating adjacency matrix from list of monster names.'''
    result = bridges.names_to_labeled_adjacency_matrix(sample_monster_names)
    np.testing.assert_array_equal(result, sample_labeled_adjacency_matrix)

def test_names_to_labeled_adjacency_matrix_squared(sample_monster_names, sample_labeled_adjacency_matrix_squared):
    '''Test generating squared adjacency matrix from list of monster names.'''
    result = bridges.names_to_labeled_adjacency_matrix(sample_monster_names, squared=True)
    np.testing.assert_array_equal(result, sample_labeled_adjacency_matrix_squared)

# test ydk_to_labeled_adjacency_matrix

def test_ydk_to_labeled_adjacency_matrix(ydk_file_path, sample_labeled_adjacency_matrix):
    '''Test generating adjacency matrix from list of monster names.'''
    result = bridges.ydk_to_labeled_adjacency_matrix(ydk_file_path)
    np.testing.assert_array_equal(result, sample_labeled_adjacency_matrix)

def test_ydk_to_labeled_adjacency_matrix_squared(ydk_file_path, sample_labeled_adjacency_matrix_squared):
    '''Test generating squared adjacency matrix from list of monster names.'''
    result = bridges.ydk_to_labeled_adjacency_matrix(ydk_file_path, squared=True)
    np.testing.assert_array_equal(result, sample_labeled_adjacency_matrix_squared)

# test find_best_bridges.
# The best bridges could change with newly released cards, so certian properties are tested

def test_length_find_best_bridges(sample_monster_names):
    '''Checks that length of best bridges is sufficiently large.'''
    result = bridges.find_best_bridges(sample_monster_names)
    assert len(result) > 4000

def test_number_of_connections(sample_monster_names):
    '''
    Checks that the most number of connections is equal to the number of monsters.
    This test only works if there is a small world bridge that connects to all monsters in sample_monster_names.
    '''
    result = bridges.find_best_bridges(sample_monster_names)
    num_monsters = len(sample_monster_names)
    assert result['number_of_connections'].max() == num_monsters

def test_bridge_score(sample_monster_names):
    '''
    Checks that the best bridge score is 1.0.
    This test only works if there is a small world bridge that connects to all monsters in sample_monster_names.
    '''
    result = bridges.find_best_bridges(sample_monster_names)
    assert result['bridge_score'].max() == 1.0

def test_all_required_target_names(sample_monster_names):
    '''
    Tests that smallest number of connections with all required targets is equal to the number of monsters.
    This test only works if there is a small world bridge that connects to all monsters in sample_monster_names.
    '''
    result = bridges.find_best_bridges(sample_monster_names, required_target_ids=sample_monster_names)
    num_required_targets = len(sample_monster_names)
    assert result['number_of_connections'].min() == num_required_targets

def test_some_required_target_names(sample_monster_names):
    '''Tests that smallest number of connections with all required targets is equal to num_in_sublist.'''
    num_in_sublist = 2
    required_target_names = sample_monster_names[:num_in_sublist]
    result = bridges.find_best_bridges(sample_monster_names, required_target_ids=required_target_names)
    assert result['number_of_connections'].min() == num_in_sublist
