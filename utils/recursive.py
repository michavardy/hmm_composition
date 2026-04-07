from __future__ import annotations

from collections.abc import Callable

import numpy as np


def solve_recursive(
    num_rows: int,
    row_shape: tuple[int, ...],
    initialize_row: Callable[[np.ndarray, int], np.ndarray],
    update_row: Callable[[np.ndarray, int], np.ndarray],
    *,
    dtype: np.dtype | type[np.floating] = np.float64,
    fill_value: float = 0.0,
    reverse: bool = False,
) -> np.ndarray:
    table = np.full((num_rows, *row_shape), fill_value, dtype=dtype)
    if num_rows == 0:
        return table

    if reverse:
        start_index = num_rows - 1
        table[start_index] = initialize_row(table, start_index)
        for row_index in range(num_rows - 2, -1, -1):
            table[row_index] = update_row(table, row_index)
    else:
        start_index = 0
        table[start_index] = initialize_row(table, start_index)
        for row_index in range(1, num_rows):
            table[row_index] = update_row(table, row_index)

    return table