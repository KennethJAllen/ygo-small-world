"""Reference calculations and edge cases for deck analysis."""

import numpy as np
import pandas as pd
import pytest

from ygo_small_world.bridges import Bridges, Deck
from ygo_small_world.connections import connections


def reference_connections(left, right):
    columns = ['type', 'attribute', 'level', 'atk', 'def']
    a, b = left[columns].to_numpy(), right[columns].to_numpy()
    return ((a[:, None, :] == b[None, :, :]).sum(axis=2) == 1).astype(np.int64)


def test_connections_and_lazy_public_matrix(synthetic_cards):
    cards = synthetic_cards
    frame = cards.get_df()
    expected = reference_connections(frame, frame)
    assert cards._adjacency_matrix is None
    np.testing.assert_array_equal(connections(frame, frame), expected)
    labeled = cards.get_labeled_adjacency_matrix()
    names = frame['name'].tolist()
    pd.testing.assert_frame_equal(labeled, pd.DataFrame(expected, index=names, columns=names))
    assert cards.get_adjacency_matrix() is cards.get_adjacency_matrix()


def test_normal_analysis_and_rankings_do_not_materialize_global_matrix(synthetic_cards):
    cards = synthetic_cards
    original = cards.get_df().copy(deep=True)
    reference = reference_connections(original, original)
    deck = Deck(cards, card_ids=list(range(1, 9)))
    bridges = Bridges(deck, cards)
    assert len(bridges) == len(bridges.get_df())
    assert len(bridges) == len(bridges.get_df())
    expected = original.copy()
    expected.insert(2, 'num_connections', reference.sum(axis=0))
    pd.testing.assert_frame_equal(cards.top_bridges(), expected.sort_values('num_connections', ascending=False).head(10))
    pd.testing.assert_frame_equal(cards.top_bridges(reverse=True), expected.sort_values('num_connections').head(10))
    expected_ids = original.loc[reference[[1, 2]].all(axis=0), 'id']
    pd.testing.assert_series_equal(cards.filter_required_targets([1, 2, 1]), expected_ids)
    pd.testing.assert_frame_equal(cards.get_df(), original)
    assert cards._adjacency_matrix is None


def test_scores_match_direct_squared_matrix_across_batches(synthetic_cards):
    deck = Deck(synthetic_cards, card_ids=list(range(1, 9)))
    bridges = Bridges(deck, synthetic_cards)
    a = reference_connections(deck.get_df(), deck.get_df())
    x = reference_connections(deck.get_df(), synthetic_cards.get_df())
    expected = []
    for i in range(x.shape[1]):
        full = np.block([[a, x[:, i:i+1]], [x[:, i:i+1].T, np.zeros((1, 1), dtype=np.int64)]])
        expected.append(np.count_nonzero(full @ full) / full.size)
    np.testing.assert_allclose(bridges._calculate_bridge_scores(), expected, rtol=0, atol=0)
    assert not x[:, 0].any()
    assert 'Card 000' not in bridges.get_df()['name'].tolist()


def test_connection_counts_do_not_overflow():
    size = 270
    frame = pd.DataFrame({'type': ['Fiend'] * size, 'attribute': [str(i) for i in range(size)],
                          'level': range(size), 'atk': range(size), 'def': range(size)})
    a = connections(frame, frame)
    squared = a @ a
    assert squared[0, 0] == size - 1
    assert squared[0, 1] == size - 2


@pytest.mark.parametrize('targets', [[999999], [1, 999999]])
def test_unknown_targets_fail(synthetic_cards, targets):
    deck = Deck(synthetic_cards, card_ids=[1, 2])
    with pytest.raises(ValueError, match='999999'):
        Bridges(deck, synthetic_cards, target_ids=targets)


def test_empty_duplicate_and_impossible_targets(synthetic_cards):
    cards = synthetic_cards
    deck = Deck(cards, card_ids=[1, 2])
    pd.testing.assert_frame_equal(Bridges(deck, cards).get_df(), Bridges(deck, cards, target_ids=[]).get_df())
    pd.testing.assert_frame_equal(Bridges(deck, cards, target_ids=[1]).get_df(), Bridges(deck, cards, target_ids=[1, 1]).get_df())
    impossible = Bridges(deck, cards, target_ids=[0])
    assert len(impossible) == 0
    assert list(impossible.get_df().columns) == list(Bridges(deck, cards).get_df().columns)
    assert impossible.get_df(top=10).empty


def test_deck_and_target_files_ignore_side_deck(synthetic_cards, tmp_path):
    path = tmp_path / 'deck.ydk'
    path.write_text('#main\n1\n2\n#extra\n0\n!side\n999999\n')
    deck = Deck(synthetic_cards, ydk_path=path)
    assert deck.get_df()['id'].tolist() == [1, 2]
    result = Bridges(deck, synthetic_cards, target_ydk_path=path).get_df()
    pd.testing.assert_frame_equal(result, Bridges(deck, synthetic_cards, target_ids=[1, 2]).get_df())


@pytest.mark.parametrize('ids', [[], [999999]])
def test_deck_requires_supported_monsters(synthetic_cards, ids):
    with pytest.raises(ValueError, match='no supported main-deck monsters'):
        Deck(synthetic_cards, card_ids=ids)


def test_ordinary_deck_still_filters_unsupported_ids(synthetic_cards):
    deck = Deck(synthetic_cards, card_ids=[1, 999999, 1])
    assert deck.get_df()['id'].tolist() == [1]
