"""Exercise the complete CLI using local card images."""

import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from ygo_small_world import utils
from ygo_small_world.cli import cli


@pytest.mark.parametrize('output_arg', [None, 'chosen/nested'])
def test_cli_outputs_in_working_directory(synthetic_cards, monkeypatch, tmp_path, output_arg):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / 'input.ydk'
    path.write_text('#main\n1\n2\n!side\n0\n')
    monkeypatch.setattr(utils, 'load_images', lambda urls: [np.zeros((624, 624, 3), dtype=np.uint8) for _ in urls])
    args = ['sw', str(path)]
    if output_arg:
        args.extend(['--output', output_arg])
    monkeypatch.setattr(sys, 'argv', args)
    cli()
    output = tmp_path / (output_arg or 'output')
    assert {path.name for path in output.iterdir()} == {'bridges.csv', 'graph.png', 'adjacency_matrix.png', 'squared_adjacency_matrix.png'}
    assert not pd.read_csv(output / 'bridges.csv').empty
    for image_path in output.glob('*.png'):
        with Image.open(image_path) as image:
            image.verify()
