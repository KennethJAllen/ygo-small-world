"""pytest fixtures."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from ygo_small_world.bridges import AllCards, Deck, Bridges

@pytest.fixture(name='ydk_file_path')
def fixture_ydk_file_path() -> Path:
    """The path for the test .ydk file."""
    ydk_file_path = Path(__file__).parent / 'test_data' / 'test_deck.ydk'
    return ydk_file_path

@pytest.fixture(name='all_cards', scope="session")
def fixture_all_cards() -> AllCards:
    """Fixture for AllCards."""
    all_cards = AllCards()
    return all_cards

@pytest.fixture(name="deck")
def fixture_deck(all_cards: AllCards, ydk_file_path: Path) -> Deck:
    """Deck for test .ydk file."""
    deck = Deck(all_cards, ydk_file_path)
    return deck

@pytest.fixture(name='bridges')
def fixture_bridges(deck: Deck, all_cards: AllCards) -> Bridges:
    """Bridges for test deck."""
    bridges = Bridges(deck, all_cards)
    return bridges

@pytest.fixture(name='bridges_all_required')
def fixture_bridges_all_required(deck: Deck, all_cards: AllCards, ydk_file_path: Path) -> Bridges:
    """Bridges for test deck, with all cards being required targets."""
    bridges = Bridges(deck, all_cards, target_ydk_path=ydk_file_path)
    return bridges

@pytest.fixture(name='bridges_some_required')
def fixture_bridges_some_required(deck: Deck, all_cards: AllCards) -> Bridges:
    """Bridges for test deck, where some cards are required targets."""
    ids = [57624336, 14558127]
    bridges = Bridges(deck, all_cards, target_ids=ids)
    return bridges

@pytest.fixture(name='sample_df')
def fixture_sample_df() -> pd.DataFrame:
    """A sample dataframe for testing the sub_df function."""
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'd'],
        'C': [10, 20, 30, 40, 50]
    })

# Data corresponding to test .ydk file

@pytest.fixture(name='card_names')
def fixture_card_names():
    """Sample card names"""
    card_names = ['Archfiend Eccentrick', 'Ash Blossom & Joyous Spring', 'Effect Veiler', 'PSY-Framegear Gamma']
    return card_names

@pytest.fixture(name='card_df')
def fixture_card_df(card_names: list[str]):
    """Sample card DataFrame."""
    return pd.DataFrame({
        'id': [57624336, 14558127, 97268402, 38814750],
        'name': card_names,
        'type': ['Fiend', 'Zombie', 'Spellcaster', 'Psychic'],
        'attribute': ['LIGHT', 'FIRE', 'LIGHT', 'LIGHT'],
        'level': [3.0, 3.0, 1.0, 2.0],
        'atk': [800.0, 0.0, 0.0, 1000.0],
        'def': [1000.0, 1800.0, 0.0, 0.0]
    })

@pytest.fixture(name='adjacency_matrix')
def fixture_adjacency_matrix():
    """A sample adjacency matrix"""
    return np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])

@pytest.fixture(name='adjacency_matrix_squared')
def fixture_adjacency_matrix_squared():
    """A sample adjacency matrix squared"""
    return np.array([[3, 1, 1, 0], [1, 2, 1, 1], [1, 1, 2, 1], [0, 1, 1, 1]])

@pytest.fixture(name='labeled_adjacency_matrix')
def fixture_labeled_adjacency_matrix(adjacency_matrix: np.ndarray, card_names):
    """A sample labeled adjacency matrix"""
    return pd.DataFrame(adjacency_matrix, index=card_names, columns=card_names)

@pytest.fixture(name='labeled_adjacency_matrix_squared')
def fixture_labeled_adjacency_matrix_squared(adjacency_matrix_squared, card_names):
    """A sample labeled adjacency matrix squared"""
    return pd.DataFrame(adjacency_matrix_squared, index=card_names, columns=card_names)
