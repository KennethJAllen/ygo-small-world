"""Plot validation and image assembly without network requests."""

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pytest

from ygo_small_world.bridges import Deck
from ygo_small_world.config import Settings
from ygo_small_world.plots import _create_matrix_img, _create_matrix_subimage, graph_fig, matrix_fig


def test_custom_settings_and_image_axes():
    settings = Settings(2, 200)
    images = [np.full((2, 2, 3), 10, dtype=np.uint8), np.full((2, 2, 3), 20, dtype=np.uint8)]
    result = _create_matrix_img(np.array([[0, 1], [1, 0]]), images, settings)
    assert result.shape == (6, 6, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result[:2, 2:4], images[0])
    np.testing.assert_array_equal(result[4:6, :2], images[1])
    assert (result[2:4, 2:4] == 200).all()
    assert (result[2:4, 4:6] == 0).all()


def test_disconnected_matrix_is_white():
    result = _create_matrix_img(np.zeros((2, 2)), [np.zeros((2, 2, 3))] * 2, Settings(2, 255))
    assert (result[2:, 2:] == 255).all()


@pytest.mark.parametrize('matrix, message', [(np.array([]), 'square'), (np.zeros((2, 3)), 'square'), (np.zeros((0, 0)), 'empty deck')])
def test_invalid_matrices(matrix, message):
    with pytest.raises(ValueError, match=message):
        _create_matrix_img(matrix, [])
    with pytest.raises(ValueError, match=message):
        _create_matrix_subimage(matrix)


def test_image_count_mismatch():
    with pytest.raises(ValueError, match='number of card images'):
        _create_matrix_img(np.zeros((2, 2)), [np.zeros((2, 2, 3))])


def test_disconnected_deck_plots(synthetic_cards, monkeypatch, tmp_path):
    from ygo_small_world import utils
    monkeypatch.setattr(utils, 'load_images', lambda urls: [np.zeros((624, 624, 3), dtype=np.uint8) for _ in urls])
    deck = Deck(synthetic_cards, card_ids=[0])
    for name, plot in [('graph', graph_fig), ('matrix', matrix_fig)]:
        path = tmp_path / f'{name}.png'
        fig = plot(deck, save_path=path)
        plt.close(fig)
        assert path.stat().st_size > 0


@pytest.mark.parametrize('plot', [graph_fig, matrix_fig])
def test_empty_deck_plot_error(synthetic_cards, plot):
    deck = Deck(synthetic_cards, card_ids=[0])
    deck._df = deck._df.iloc[:0]
    deck._adjacency_matrix = np.empty((0, 0), dtype=np.int64)
    with pytest.raises(ValueError, match='empty deck'):
        plot(deck)
