'''Tests for graph_adjacency_visualizer.py'''

import os
import pytest
from ygo_small_world import graph_adjacency_visualizer as gav


@pytest.fixture
def sample_card_names():
    '''A list sample card names.'''
    return ['Archfiend Eccentrick', 'Ash Blossom & Joyous Spring', 'Effect Veiler', 'PSY-Framegear Gamma']

@pytest.fixture
def ydk_file_path():
    '''The path of the test ydk file.'''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ydk_file = 'test_deck.ydk'
    ydk_file_path = os.path.join(current_dir, ydk_file)
    return ydk_file_path

# Test names_to_image_urls

def test_names_to_image_urls(sample_card_names):
    '''Test getting image urls from card names.'''
    result = gav.names_to_image_urls(sample_card_names)
    expected = [
        'https://images.ygoprodeck.com/images/cards_cropped/57624336.jpg',
        'https://images.ygoprodeck.com/images/cards_cropped/14558127.jpg',
        'https://images.ygoprodeck.com/images/cards_cropped/97268402.jpg',
        'https://images.ygoprodeck.com/images/cards_cropped/38814750.jpg']
    assert result == expected