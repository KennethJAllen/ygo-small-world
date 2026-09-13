"""Tests for small_world_bridge_generator.py"""

import numpy as np
import pandas as pd
import networkx as nx
from ygo_small_world.bridges import AllCards, Deck, Bridges

# test AllCards

def test_all_cards_populated(all_cards: AllCards):
    """Test there are no nulls"""
    assert len(all_cards.get_df()) > 6000

# test Deck

def test_deck_df(deck: Deck, card_df: pd.DataFrame):
    """Test getting df from sample monster names'''"""
    df = deck.get_df().reset_index(drop=True)
    small_world_columns = ['id', 'name', 'type', 'attribute', 'level', 'atk', 'def']
    result = df[small_world_columns].sort_values('id').reset_index(drop=True)
    pd.testing.assert_frame_equal(result, card_df.sort_values('id').reset_index(drop=True))

def test_deck_adjacency_matrix(deck: Deck, adjacency_matrix: np.ndarray, card_names):
    """Test generating adjacency matrix from dataframe"""
    result = deck.get_adjacency_matrix()
    order = [deck.get_df()['name'].tolist().index(name) for name in card_names]
    result = result[np.ix_(order, order)]
    np.testing.assert_array_equal(result, adjacency_matrix)

def test_deck_adjacency_matrix_squared(deck: Deck, adjacency_matrix_squared: np.ndarray, card_names):
    """Test generating squared adjacency matrix from dataframe"""
    result = deck.get_adjacency_matrix(squared=True)
    order = [deck.get_df()['name'].tolist().index(name) for name in card_names]
    result = result[np.ix_(order, order)]
    np.testing.assert_array_equal(result, adjacency_matrix_squared)

def test_deck_labeled_adjacency_matrix(deck: Deck, labeled_adjacency_matrix: np.ndarray):
    """Test generating adjacency matrix from list of monster names"""
    result = deck.get_labeled_adjacency_matrix()
    pd.testing.assert_frame_equal(result.loc[labeled_adjacency_matrix.index, labeled_adjacency_matrix.columns],
                                  labeled_adjacency_matrix)

def test_deck_labeled_adjacency_matrix_squared(deck: Deck, labeled_adjacency_matrix_squared: np.ndarray):
    """Test generating squared adjacency matrix from list of monster names"""
    result = deck.get_labeled_adjacency_matrix(squared=True)
    pd.testing.assert_frame_equal(result.loc[labeled_adjacency_matrix_squared.index, labeled_adjacency_matrix_squared.columns],
                                  labeled_adjacency_matrix_squared)

def test_graph(deck: Deck):
    """Test graph has same adjacency matrix."""
    graph = deck.get_graph()
    result = nx.to_numpy_array(graph)
    np.testing.assert_array_equal(result, deck.get_adjacency_matrix())

# test Bridges

def test_length_find_best_bridges(bridges: Bridges):
    """Checks that length of best bridges is sufficiently large"""
    result = bridges.get_df()
    assert len(result) > 4000

def test_number_of_connections(bridges: Bridges, card_names: list[str]):
    """
    Checks that the most number of connections is equal to the number of monsters.
    This test only works if there is a small world bridge that connects to all monsters in sample_monster_names.
    """
    result = bridges.get_df()
    num_monsters = len(card_names)
    assert result['number_of_connections'].max() == num_monsters

def test_bridge_score(bridges: Bridges):
    """
    Checks that the best bridge score is 1.0.
    This test only works if there is a small world bridge that connects to all monsters in sample_monster_names.
    """
    result = bridges.get_df()
    assert result['bridge_score'].max() == 1.0

def test_all_required_target_names(bridges_all_required: Bridges, card_names: list[str]):
    """
    Tests that smallest number of connections with all required targets is equal to the number of monsters.
    This test only works if there is a small world bridge that connects to all monsters in sample_monster_names.
    """
    result = bridges_all_required.get_df()
    num_required_targets = len(card_names)
    assert result['number_of_connections'].min() == num_required_targets

def test_some_required_target_names(bridges_some_required: Bridges):
    '''Tests that smallest number of connections with all required targets is equal to num_in_sublist.'''
    result = bridges_some_required.get_df()
    assert result['number_of_connections'].min() == 2
