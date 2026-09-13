"""Database locations and safe updates, with all network traffic mocked."""

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from ygo_small_world import data_paths, update_data
from ygo_small_world.bridges import AllCards


@pytest.fixture
def api_card():
    return {'id': 1, 'name': 'Test monster', 'type': 'Effect Monster', 'race': 'Fiend',
            'attribute': 'DARK', 'level': 3, 'atk': -1, 'def': 100,
            'frameType': 'effect', 'card_images': [{'image_url_cropped': 'https://example.invalid/1.jpg'}]}


def test_source_database_independent_of_cwd(monkeypatch, tmp_path):
    expected = data_paths.readable_card_path()
    monkeypatch.chdir(tmp_path)
    assert data_paths.readable_card_path() == expected
    assert len(AllCards()) > 6000


def test_installed_database_fallback_and_override(monkeypatch, tmp_path):
    monkeypatch.setattr(data_paths, '_source_root', lambda: None)
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    bundled = Path(data_paths.__file__).parent / 'data' / 'cardinfo.csv'
    assert data_paths.readable_card_path() == bundled
    writable = data_paths.writable_card_path()
    assert writable == tmp_path / '.ygo-small-world' / 'cardinfo.csv'
    writable.parent.mkdir()
    writable.touch()
    assert data_paths.readable_card_path() == writable


def test_update_writes_valid_data_and_creates_directory(monkeypatch, tmp_path, api_card):
    output = tmp_path / 'new' / 'cardinfo.csv'
    monkeypatch.setattr(update_data, 'writable_card_path', lambda: output)
    api_card['name'] = 'Monster, "quoted"\n\u00e9'
    response = Mock()
    response.json.return_value = {'data': [dict(api_card, id=2, name='NA'), api_card]}
    monkeypatch.setattr(update_data.requests, 'get', Mock(return_value=response))
    update_data.update_card_data()
    response.raise_for_status.assert_called_once()
    frame = pd.read_csv(output)
    assert frame.iloc[0]['type'] == 'Fiend'
    assert frame.iloc[0]['atk'] == -1
    assert frame['id'].tolist() == [1, 2]
    assert frame.iloc[0]['name'] == api_card['name']
    assert frame.columns.tolist() == ['id', 'name', 'type', 'attribute', 'level', 'atk', 'def', 'img_url']
    monkeypatch.setattr('ygo_small_world.bridges.readable_card_path', lambda: output)
    loaded = AllCards()._df
    assert loaded['name'].tolist() == [api_card['name'], 'NA']
    pd.testing.assert_frame_equal(loaded, pd.read_csv(output, keep_default_na=False))
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize('failure', ['http', 'timeout', 'json', 'empty', 'missing_property', 'null', 'duplicates', 'write', 'replace'])
def test_failed_update_preserves_previous_database(monkeypatch, tmp_path, api_card, failure):
    output = tmp_path / 'cardinfo.csv'
    pd.DataFrame({'original': [1]}).to_csv(output, index=False)
    original = output.read_bytes()
    monkeypatch.setattr(update_data, 'writable_card_path', lambda: output)
    response = Mock()
    payload = {'data': [api_card]}
    if failure == 'http':
        response.raise_for_status.side_effect = requests.HTTPError('failed')
    elif failure == 'json':
        response.json.side_effect = ValueError('invalid JSON')
    elif failure == 'empty':
        payload = {'data': []}
    elif failure == 'missing_property':
        del api_card['attribute']
    elif failure == 'null':
        api_card['atk'] = None
    elif failure == 'duplicates':
        payload['data'].append(api_card.copy())
    elif failure == 'write':
        monkeypatch.setattr(pd.DataFrame, 'to_csv', Mock(side_effect=OSError('disk full')))
    elif failure == 'replace':
        monkeypatch.setattr(update_data.os, 'replace', Mock(side_effect=OSError('disk error')))
    response.json.return_value = payload
    get = Mock(return_value=response)
    if failure == 'timeout':
        get.side_effect = requests.Timeout('timeout')
    monkeypatch.setattr(update_data.requests, 'get', get)
    with pytest.raises((ValueError, KeyError, OSError, requests.RequestException)):
        update_data.update_card_data()
    assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]
