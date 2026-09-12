"""Calculate Small World connections with bounded comparison temporaries."""

import numpy as np
import pandas as pd

BLOCK_SIZE = 256
PROPERTIES = ['type', 'attribute', 'level', 'atk', 'def']


def connection_blocks(left: pd.DataFrame, right: pd.DataFrame):
    """Yield (row offset, integer connections) for blocks of left-hand cards."""
    for frame in (left, right):
        if not all(column in frame.columns for column in PROPERTIES):
            raise ValueError(f"DataFrame must have columns: {', '.join(PROPERTIES)}")
    left_values = left[PROPERTIES].to_numpy()
    right_values = right[PROPERTIES].to_numpy()
    for start in range(0, len(left), BLOCK_SIZE):
        block = left_values[start:start + BLOCK_SIZE]
        matches = np.zeros((len(block), len(right)), dtype=np.uint8)
        for column in range(len(PROPERTIES)):
            matches += block[:, column, None] == right_values[None, :, column]
        yield start, (matches == 1).astype(np.int64)


def connections(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    """Return integer connections without allocating a global card matrix."""
    result = np.empty((len(left), len(right)), dtype=np.int64)
    for start, block in connection_blocks(left, right):
        result[start:start + len(block)] = block
    return result
