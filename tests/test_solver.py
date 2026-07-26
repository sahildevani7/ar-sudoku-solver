import numpy as np
import pytest

from sudoku_ar.solver import solve_wrapper

# A well-known puzzle/solution pair (from Project Euler #96, puzzle 1).
PUZZLE = np.array([
    [0, 0, 3, 0, 2, 0, 6, 0, 0],
    [9, 0, 0, 3, 0, 5, 0, 0, 1],
    [0, 0, 1, 8, 0, 6, 4, 0, 0],
    [0, 0, 8, 1, 0, 2, 9, 0, 0],
    [7, 0, 0, 0, 0, 0, 0, 0, 8],
    [0, 0, 6, 7, 0, 8, 2, 0, 0],
    [0, 0, 2, 6, 0, 9, 5, 0, 0],
    [8, 0, 0, 2, 0, 3, 0, 0, 9],
    [0, 0, 5, 0, 1, 0, 3, 0, 0],
], dtype=np.uint8)

SOLUTION = np.array([
    [4, 8, 3, 9, 2, 1, 6, 5, 7],
    [9, 6, 7, 3, 4, 5, 8, 2, 1],
    [2, 5, 1, 8, 7, 6, 4, 9, 3],
    [5, 4, 8, 1, 3, 2, 9, 7, 6],
    [7, 2, 9, 5, 6, 4, 1, 3, 8],
    [1, 3, 6, 7, 9, 8, 2, 4, 5],
    [3, 7, 2, 6, 8, 9, 5, 1, 4],
    [8, 1, 4, 2, 5, 3, 7, 6, 9],
    [6, 9, 5, 4, 1, 7, 3, 8, 2],
], dtype=np.uint8)

# Two cells swapped versus SOLUTION in the same row -> no valid completion exists.
UNSOLVABLE = PUZZLE.copy()
UNSOLVABLE[0, 0] = 4
UNSOLVABLE[0, 1] = 4


def test_solves_known_puzzle():
    result, message = solve_wrapper(PUZZLE.copy())
    assert result is not None
    assert np.array_equal(result, SOLUTION)
    assert "Solved" in message


def test_preserves_given_digits():
    result, _ = solve_wrapper(PUZZLE.copy())
    given = PUZZLE != 0
    assert np.array_equal(result[given], PUZZLE[given])


def test_unsolvable_puzzle_returns_none():
    result, message = solve_wrapper(UNSOLVABLE.copy())
    assert result is None
    assert message is None


def test_second_known_puzzle():
    # A different givens layout than PUZZLE, masked out of SOLUTION, to make sure
    # the solver isn't just tuned to one fixture. Note: this exact-cover implementation's
    # runtime is sensitive to clue count/placement (a fully empty grid, or some sparse
    # ~20-clue layouts, can take from seconds to effectively forever in pure Python) -
    # so this intentionally keeps enough givens (matching realistic puzzle difficulty)
    # to stay fast and deterministic as a test.
    rng = np.random.RandomState(7)
    mask = np.zeros(81, dtype=bool)
    mask[rng.choice(81, size=30, replace=False)] = True
    grid = np.where(mask.reshape(9, 9), SOLUTION, 0).astype(np.uint8)

    result, message = solve_wrapper(grid)
    assert result is not None
    given = grid != 0
    assert np.array_equal(result[given], grid[given])
    assert np.all(result >= 1) and np.all(result <= 9)
