'''
End-to-end recognition check against a real camera capture (a puzzle displayed on a phone
screen, photographed by the webcam) rather than a synthetic drawing.

The synthetic fixtures elsewhere in this suite passed happily while the app failed on real
input, so this pins the behaviour that actually mattered: on this frame the reader originally
found 73 of 81 cells "occupied" with a worst-cell confidence of 0.21 and produced an invalid
grid. Grid-line removal plus centring each digit brings that to 26 givens at 0.91.
'''
import os

import cv2
import numpy as np
import pytest

from sudoku_ar import config
from sudoku_ar.recognizer import DigitRecognizer
from sudoku_ar.validator import isValidConfig

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "real_capture_warped_inv.png")

# What a human reads off the captured frame. Cell [2][0] is blank in the real puzzle - a streak
# of screen glare sits there, which the reader still calls a "1"; see the xfail below.
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
    return recognizer.extract_digit(warped_inv)


def test_finds_a_plausible_number_of_givens(read):
    digits, _, _ = read
    # the old reader saw 73 here, because ruled lines counted as ink
    assert 20 <= np.count_nonzero(digits) <= 30


def test_read_is_confident_enough_to_accept(read):
    _, confident, min_confidence = read
    assert min_confidence > config.MIN_CONFIDENCE
    assert confident is True


def test_read_obeys_sudoku_rules(read):
    digits, _, _ = read
    assert isValidConfig(digits)


def test_almost_every_cell_matches_the_human_reading(read):
    digits, _, _ = read
    wrong = np.count_nonzero(digits != EXPECTED)
    assert wrong <= 1, "regressed to %d wrong cells:\n%s" % (wrong, digits)


def test_blank_cells_are_not_invented(read):
    digits, _, _ = read
    # every cell the reader calls blank really is blank, and vice versa apart from the glare cell
    disagreements = list(zip(*np.where((digits == 0) != (EXPECTED == 0))))
    assert disagreements in ([], [(2, 0)]), disagreements


@pytest.mark.xfail(strict=True, reason="glare streak on the phone screen at [2][0] still reads as a 1")
def test_glare_streak_is_not_read_as_a_digit(read):
    digits, _, _ = read
    assert digits[2][0] == 0
