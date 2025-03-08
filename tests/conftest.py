"""pytest fixtures."""

from pathlib import Path
import pytest
import pandas as pd
from ygo_small_world.bridges import AllCards, Deck, Bridges

@pytest.fixture(name='ydk_file_path')
def fixture_ydk_file_path() -> Path:
    """The path for the test .ydk file."""
    ydk_file_path = Path(__file__).parent / 'test_data' / 'test_deck.ydk'
    return ydk_file_path

@pytest.fixture(name='all_cards')
def fixture_all_cards() -> AllCards:
    """Fixture for AllCards."""
    all_cards = AllCards()
    return all_cards

@pytest.fixture(name="deck")
def fixture_deck(all_cards, ydk_file_path) -> Deck:
    """Deck for test .ydk file."""
    deck = Deck(all_cards, ydk_file_path)
    return deck

@pytest.fixture(name='bridges')
def fixture_bridges(deck, all_cards) -> Bridges:
    """Bridges for test deck."""
    bridges = Bridges(deck, all_cards)
    return bridges

@pytest.fixture(name='sample_df')
def fixture_sample_df() -> pd.DataFrame:
    '''A sample dataframe for testing the sub_df function.'''
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'd'],
        'C': [10, 20, 30, 40, 50]
    })
