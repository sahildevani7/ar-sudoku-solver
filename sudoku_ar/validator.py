import numpy as np


def _no_duplicates(groups):
    '''
        groups: (9, 9) array where each row holds one line of cells to check (a sudoku row,
        column, or 3x3 box). Returns False if any nonzero value repeats within a single group.
    '''
    sorted_groups = np.sort(groups, axis=1)
    adjacent_equal = (sorted_groups[:, 1:] == sorted_groups[:, :-1]) & (sorted_groups[:, 1:] != 0)
    return not adjacent_equal.any()


def isValidConfig(arr):
    '''
        Checks every row, column, and 3x3 box for duplicate (nonzero) digits.
    '''
    arr = np.asarray(arr)
    boxes = arr.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
    return bool(_no_duplicates(arr) and _no_duplicates(arr.T) and _no_duplicates(boxes))
