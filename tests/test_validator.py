import numpy as np

from grid_validator import isValidConfig

from tests.test_solver import PUZZLE, SOLUTION


def test_empty_grid_is_valid():
    assert isValidConfig(np.zeros((9, 9), dtype=np.uint8))


def test_solved_puzzle_is_valid():
    assert isValidConfig(SOLUTION)


def test_partial_puzzle_is_valid():
    assert isValidConfig(PUZZLE)


def test_duplicate_in_row_is_invalid():
    grid = SOLUTION.copy()
    grid[0, 1] = grid[0, 0]
    assert not isValidConfig(grid)


def test_duplicate_in_column_is_invalid():
    grid = SOLUTION.copy()
    grid[1, 0] = grid[0, 0]
    assert not isValidConfig(grid)


def test_duplicate_in_box_is_invalid():
    grid = SOLUTION.copy()
    grid[1, 1] = grid[0, 0]
    assert not isValidConfig(grid)
