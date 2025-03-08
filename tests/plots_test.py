'''Tests for graph_adjacency_visualizer.py'''

from pathlib import Path
import pytest
from PIL import Image
import numpy as np
import networkx as nx
from ygo_small_world import plots as gav
from ygo_small_world import bridges as sw


@pytest.fixture(name='sample_card_names')
def fixture_sample_card_names():
    '''A list sample card names.'''
    return ['Archfiend Eccentrick',
            'Ash Blossom & Joyous Spring',
            'Effect Veiler',
            'PSY-Framegear Gamma']

@pytest.fixture(name='ydk_file_path')
def fixture_ydk_file_path():
    """The path of the test ydk file."""
    ydk_file_path = Path(__file__).parent / 'test_data' / 'test_deck.ydk'
    return ydk_file_path

# test names_to_graph

def test_names_to_graph(sample_card_names):
    '''Test generating graph from list of names by comparing adjacency matrices'''
    graph = gav.names_to_graph(sample_card_names)
    array = nx.to_numpy_array(graph)
    expected_array = sw.card_ids_to_adjacency_matrix(sample_card_names)
    assert np.array_equal(array, expected_array)

# test ydk_to_graph

def test_ydk_to_graph(ydk_file_path):
    '''Test generating graph from list of names by comparing adjacency matrices'''
    graph = gav.ydk_to_graph(ydk_file_path)
    array = nx.to_numpy_array(graph)
    expected_array = sw.ydk_to_adjacency_matrix(ydk_file_path)
    assert np.array_equal(array, expected_array)

# test names_to_image_urls

def test_names_to_actual_image_urls(sample_card_names):
    '''Test getting image urls from card names.'''
    result = gav.names_to_image_urls(sample_card_names)
    expected = [
        'https://images.ygoprodeck.com/images/cards_cropped/57624336.jpg',
        'https://images.ygoprodeck.com/images/cards_cropped/14558127.jpg',
        'https://images.ygoprodeck.com/images/cards_cropped/97268402.jpg',
        'https://images.ygoprodeck.com/images/cards_cropped/38814750.jpg'
        ]
    assert result == expected

# test names_to_matrix_image

def test_names_to_matrix_image(sample_card_names):
    '''Test generating image of matrix from sample card names.'''
    result = gav.names_to_matrix_image(sample_card_names)
    reference_image = Image.open('tests/test_images/test-matrix.png')
    reference_array = np.array(reference_image)
    assert np.array_equal(result, reference_array)

def test_names_to_matrix_image_squared(sample_card_names):
    '''Test generating image of squared matrix from sample card names.'''
    result = gav.names_to_matrix_image(sample_card_names, squared=True).astype(np.uint8)
    reference_image = Image.open('tests/test_images/test-matrix-squared.png')
    reference_array = np.array(reference_image)
    assert np.array_equal(result, reference_array)

# test ydk_to_matrix_image

def test_ydk_to_matrix_image(ydk_file_path):
    '''Test generating image of matrix from sample card names.'''
    result = gav.ydk_to_matrix_image(ydk_file_path)
    reference_image = Image.open('tests/test_images/test-matrix.png')
    reference_array = np.array(reference_image)
    assert np.array_equal(result, reference_array)

def test_ydk_to_matrix_image_squared(ydk_file_path):
    '''Test generating image of squared matrix from sample card names.'''
    result = gav.ydk_to_matrix_image(ydk_file_path, squared=True).astype(np.uint8)
    reference_image = Image.open('tests/test_images/test-matrix-squared.png')
    reference_array = np.array(reference_image)
    assert np.array_equal(result, reference_array)
