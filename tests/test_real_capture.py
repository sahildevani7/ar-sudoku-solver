'''
End-to-end recognition check against a real camera capture (a puzzle displayed on a phone
screen, photographed by the webcam) rather than a synthetic drawing.

The synthetic fixtures elsewhere in this suite passed happily while the app failed on real
input, so this pins the behaviour that actually mattered. On this frame the reader has at
various points found 73 of 81 cells "occupied" (ruled lines counted as ink), read every row
shifted up by one (cells cut at even ninths instead of at the real ruling), and invented digits
out of screen glare. It should now read the puzzle exactly.
'''
import os

import cv2
import numpy as np
import pytest

from sudoku_ar import config, detector
from sudoku_ar.recognizer import DigitRecognizer
from sudoku_ar.solver import solve_wrapper
from sudoku_ar.validator import isValidConfig

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "real_capture_warped_inv.png")

# What a human reads off the captured frame.
EXPECTED = np.array([
    [0, 3, 0, 0, 0, 0, 0, 4, 0],
    [5, 4, 0, 0, 7, 1, 0, 2, 0],
    [0, 0, 0, 3, 0, 0, 0, 0, 6],
    [0, 0, 8, 0, 0, 9, 0, 0, 0],
    [3, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 7, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 0, 8, 9],
    [7, 8, 5, 9, 0, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 0, 0, 6, 2],
], dtype=np.uint8)


@pytest.fixture(scope="module")
def read():
    recognizer = DigitRecognizer()
    warped_inv = cv2.imread(FIXTURE, cv2.IMREAD_GRAYSCALE)
    assert warped_inv is not None, "fixture image missing"
    boundaries = detector.find_cell_boundaries(warped_inv)
    assert boundaries is not None, "ruling should be detectable on this capture"
    return recognizer.extract_digit(warped_inv, boundaries)


def test_finds_the_right_number_of_givens(read):
    digits, _, _ = read
    # the reader once saw 73 here, because ruled lines counted as ink
    assert np.count_nonzero(digits) == np.count_nonzero(EXPECTED)


def test_read_is_confident_enough_to_accept(read):
    _, confident, min_confidence = read
    assert min_confidence > config.MIN_CONFIDENCE
    assert confident is True


def test_read_obeys_sudoku_rules(read):
    digits, _, _ = read
    assert isValidConfig(digits)


def test_every_cell_matches_the_human_reading(read):
    digits, _, _ = read
    assert np.array_equal(digits, EXPECTED), "read back:\n%s" % digits


def test_glare_streaks_are_not_read_as_digits(read):
    '''Column 0 carries two streaks of screen glare; neither is a digit.'''
    digits, _, _ = read
    assert digits[0][0] == 0
    assert digits[2][0] == 0


def test_the_read_puzzle_actually_solves(read):
    digits, _, _ = read
    solution, _ = solve_wrapper(digits.copy())
    assert solution is not None


def test_cell_boundaries_are_fitted_not_evenly_spaced():
    '''
    The warp clipped this puzzle's top border and left dead space below it, so the true cell
    edges are not at even ninths - assuming they were shifted every row up by one.
    '''
    warped_inv = cv2.imread(FIXTURE, cv2.IMREAD_GRAYSCALE)
    rows, cols = detector.find_cell_boundaries(warped_inv)
    height = warped_inv.shape[0]

    assert len(rows) == 10 and len(cols) == 10
    assert rows[0] < 0          # the puzzle's top edge fell outside the warp
    assert rows[-1] < height    # ...and its bottom edge inside it
    spacings = np.diff(rows)
    assert spacings.std() < 1.0  # evenly spaced among themselves, just offset
