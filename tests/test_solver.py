import time

import numpy as np
import pytest

from sudoku_ar import config
from sudoku_ar.solver import solve_wrapper
from sudoku_ar.validator import isValidConfig

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

# Consistent givens, but far too few to pin down a completion: ~6.7e21 grids satisfy this.
# Nothing here is contradictory, so the search happily runs forever unless it is bounded.
NEAR_EMPTY = np.zeros((9, 9), dtype=np.uint8)
NEAR_EMPTY[0, 0] = 1
NEAR_EMPTY[4, 4] = 5
NEAR_EMPTY[8, 8] = 9

# No duplicate among the givens, so this can't fail fast the way UNSOLVABLE does - the search has
# to exhaust. The top-left box holds 1-8, forcing its one free cell (2,2) to be 9, but column 2
# already has a 9 lower down, so no digit fits there.
NO_COMPLETION = np.zeros((9, 9), dtype=np.uint8)
NO_COMPLETION[0, :3] = [1, 2, 3]
NO_COMPLETION[1, :3] = [4, 5, 6]
NO_COMPLETION[2, :2] = [7, 8]
NO_COMPLETION[3, 2] = 9


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
    # A different givens layout than PUZZLE, masked out of SOLUTION, to make sure the solver
    # isn't just tuned to one fixture. 30 givens keeps this at realistic puzzle difficulty and
    # deterministic; sparser layouts now hit SOLVER_TIMEOUT_SECONDS rather than running away,
    # but they'd make the assertions below depend on the machine's speed.
    rng = np.random.RandomState(7)
    mask = np.zeros(81, dtype=bool)
    mask[rng.choice(81, size=30, replace=False)] = True
    grid = np.where(mask.reshape(9, 9), SOLUTION, 0).astype(np.uint8)

    # solve_wrapper fills its argument in place, so compare against a snapshot of the givens.
    result, message = solve_wrapper(grid.copy())
    assert result is not None
    given = grid != 0
    assert np.array_equal(result[given], grid[given])
    assert np.all(result >= 1) and np.all(result <= 9)


def test_near_empty_grid_returns_promptly():
    # Regression test for the hang. solve_wrapper used to materialise *every* completion before
    # taking the first, and this grid has ~6.7e21 of them, so it never returned. Taking only the
    # first makes it immediate. Note the answer here is a real solution, not (None, None): an
    # under-constrained grid has completions, and returning one is correct. What was broken was
    # the time it took, so that is what this pins down.
    start = time.monotonic()
    result, message = solve_wrapper(NEAR_EMPTY.copy())
    elapsed = time.monotonic() - start

    assert elapsed < config.SOLVER_TIMEOUT_SECONDS + 2.0
    if result is not None:
        given = NEAR_EMPTY != 0
        assert np.array_equal(result[given], NEAR_EMPTY[given])
        assert isValidConfig(result)
        assert np.all(result >= 1) and np.all(result <= 9)


def test_exhausted_budget_returns_none():
    # The deadline is what bounds a grid whose *first* solution is slow to reach - not something
    # the fixtures here can produce on demand, so drive it directly: a zero budget, checked at
    # every node, has to trip on the first one.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sudoku_ar.solver._Deadline.CHECK_INTERVAL", 1)
        result, message = solve_wrapper(PUZZLE.copy(), timeout=0.0)

    assert result is None
    assert message is None


def test_grid_without_completion_returns_none():
    result, message = solve_wrapper(NO_COMPLETION.copy())
    assert result is None
    assert message is None


def test_does_not_swallow_keyboard_interrupt():
    # The old bare `except:` turned a Ctrl-C during a long solve into a silent (None, None).
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sudoku_ar.solver.solve_sudoku", interrupt)
        with pytest.raises(KeyboardInterrupt):
            solve_wrapper(PUZZLE.copy())
