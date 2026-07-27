import time
import numpy as np

from itertools import product

from . import config


class SolverTimeout(Exception):
    """The exact-cover search ran past its budget before producing a solution."""


class _Deadline:
    """Wall-clock budget for the search.

    The clock is only read once every CHECK_INTERVAL nodes; checking it at every node
    measurably slows a search that otherwise does nothing but dict/set operations.
    """

    CHECK_INTERVAL = 512

    def __init__(self, seconds):
        self.expires_at = time.monotonic() + seconds
        self.countdown = self.CHECK_INTERVAL

    def check(self):
        self.countdown -= 1
        if self.countdown > 0:
            return
        self.countdown = self.CHECK_INTERVAL
        if time.monotonic() >= self.expires_at:
            raise SolverTimeout()


def solve_sudoku(size, grid, deadline=None):
    R, C = size
    N = R * C
    X = ([("rc", rc) for rc in product(range(N), range(N))] +
         [("rn", rn) for rn in product(range(N), range(1, N + 1))] +
         [("cn", cn) for cn in product(range(N), range(1, N + 1))] +
         [("bn", bn) for bn in product(range(N), range(1, N + 1))])

    Y = dict()
    for r, c, n in product(range(N), range(N), range(1, N + 1)):
        b = (r // R) * R + (c // C)  # Box number
        Y[(r, c, n)] = [
            ("rc", (r, c)),
            ("rn", (r, n)),
            ("cn", (c, n)),
            ("bn", (b, n))]
    X, Y = exact_cover(X, Y)
    for i, row in enumerate(grid):
        for j, n in enumerate(row):
            if n:
                select(X, Y, (i, j, n))
    for solution in solve(X, Y, [], deadline):
        for (r, c, n) in solution:
            grid[r][c] = n
        yield grid


def exact_cover(X, Y):
    X = {j: set() for j in X}
    for i, row in Y.items():
        for j in row:
            X[j].add(i)
    return X, Y


def solve(X, Y, solution, deadline=None):
    if deadline is not None:
        deadline.check()
    if not X:
        yield list(solution)
    else:
        c = min(X, key=lambda c: len(X[c]))
        for r in list(X[c]):
            solution.append(r)
            cols = select(X, Y, r)
            for s in solve(X, Y, solution, deadline):
                yield s
            deselect(X, Y, r, cols)
            solution.pop()


def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols


def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)


def solve_wrapper(arr, timeout=None):
    """Return (solution, message), or (None, None) if `arr` has no solution or takes too long.

    Note that `arr` is filled in place by the search - callers must pass a copy if they
    still need the puzzle's givens afterwards.
    """
    if timeout is None:
        timeout = config.SOLVER_TIMEOUT_SECONDS

    start = time.monotonic()
    try:
        # Only ever the first solution: an under-constrained grid (a misread) has
        # astronomically many, and enumerating them all is unbounded work.
        solution = next(solve_sudoku(size=(3, 3), grid=arr, deadline=_Deadline(timeout)), None)
    except SolverTimeout:
        return None, None
    except KeyError:
        # A given that can't be placed - a repeated digit in a row/column/box, or a value
        # outside 1-9. Either way the grid is unsolvable as read.
        return None, None

    if solution is None:  # search exhausted: no completion exists
        return None, None
    return np.array(solution, dtype=np.uint8), "Solved in %.4fs" % (time.monotonic() - start)
