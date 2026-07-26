import cv2
import numpy as np
import pytest

from sudoku_ar import config
from sudoku_ar.recognizer import DigitRecognizer, remove_border, remove_grid_lines, extract_centered_digit, empty


@pytest.fixture(scope="module")
def recognizer():
    return DigitRecognizer()


def blank_grid():
    return np.full((config.WARP_SIZE, config.WARP_SIZE), 255, dtype=np.uint8)


def test_blank_grid_returns_all_zero_and_confident(recognizer):
    result, confident, min_confidence = recognizer.extract_digit(blank_grid())
    assert result.shape == (9, 9)
    assert result.dtype == np.uint8
    assert np.all(result == 0)
    assert confident is True
    assert min_confidence == 1.0


def test_non_empty_cells_produce_in_range_digits(recognizer):
    grid = blank_grid()
    posx = posy = config.WARP_SIZE // 9
    # draw dark blobs in a few cells so empty() reports them as non-empty
    for (i, j) in [(2, 3), (5, 7), (0, 0)]:
        cx, cy = j * posx + posx // 2, i * posy + posy // 2
        cv2.circle(grid, (cx, cy), posx // 4, 0, -1)

    result, confident, min_confidence = recognizer.extract_digit(grid)
    assert result.shape == (9, 9)
    assert isinstance(confident, bool)
    assert 0.0 <= min_confidence <= 1.0
    for (i, j) in [(2, 3), (5, 7), (0, 0)]:
        assert 1 <= result[i, j] <= 9
    # cells that were never touched must stay empty
    assert result[8, 8] == 0


def test_batching_matches_a_direct_single_cell_prediction(recognizer):
    '''
    Regression test for the batching refactor: predicts one specific cell both through
    extract_digit's batched path and by replaying the same preprocessing steps as a single-item
    model call, to guard against a cell_positions/cell_batch indexing mixup.
    '''
    grid = blank_grid()
    posx = posy = config.WARP_SIZE // 9
    target_i, target_j = 4, 6
    cx, cy = target_j * posx + posx // 2, target_i * posy + posy // 2
    cv2.circle(grid, (cx, cy), posx // 4, 0, -1)

    result, _, _ = recognizer.extract_digit(grid)

    digit_size = config.DIGIT_SIZE
    cleaned = remove_grid_lines(grid)
    cell = cleaned[posy * target_i: posy * (target_i + 1), posx * target_j: posx * (target_j + 1)]
    prepared = extract_centered_digit(cell)
    assert prepared is not None

    direct_pred = recognizer.model(prepared.reshape(1, digit_size, digit_size, 1), training=False).numpy().argmax(axis=1)[0] + 1
    assert result[target_i, target_j] == direct_pred


def test_remove_grid_lines_erases_ruling_but_keeps_digits():
    '''
    The ruled lines land inside cell crops whenever the warp is slightly off, which made empty()
    report nearly every cell as occupied (73 of 81 on a real capture) and fed line fragments to
    the model. Ruling must go; digit strokes must survive.
    '''
    size = config.WARP_SIZE
    grid = np.full((size, size), 255, dtype=np.uint8)
    for i in range(10):  # full-length ruling
        offset = min(i * (size // 9), size - 1)
        cv2.line(grid, (0, offset), (size - 1, offset), 0, 3)
        cv2.line(grid, (offset, 0), (offset, size - 1), 0, 3)

    cell = size // 9
    digit_pos = (3 * cell + cell // 3, 2 * cell + 2 * cell // 3)
    cv2.putText(grid, "5", digit_pos, cv2.FONT_HERSHEY_SIMPLEX, cell / 45, 0, 3)

    ink_before = np.count_nonzero(grid < 128)
    cleaned = remove_grid_lines(grid)
    ink_after = np.count_nonzero(cleaned < 128)

    assert ink_after < ink_before * 0.5  # the ruling (most of the ink) is gone
    assert ink_after > 0                 # but the digit survived

    # and the surviving ink sits in the cell the digit was drawn in
    ys, xs = np.where(cleaned < 128)
    assert 2 <= int(np.median(ys)) // cell <= 3
    assert 3 <= int(np.median(xs)) // cell <= 4


def test_line_removal_stops_empty_cells_reading_as_occupied(recognizer):
    '''An empty grid that is nothing but ruling must yield no digits at all.'''
    size = config.WARP_SIZE
    grid = np.full((size, size), 255, dtype=np.uint8)
    for i in range(10):
        offset = min(i * (size // 9), size - 1)
        cv2.line(grid, (0, offset), (size - 1, offset), 0, 3)
        cv2.line(grid, (offset, 0), (offset, size - 1), 0, 3)

    result, _, _ = recognizer.extract_digit(grid)
    assert np.count_nonzero(result) == 0


def test_empty_helper_on_mostly_white_and_mostly_black():
    white = np.full((20, 20), 255, dtype=np.uint8)
    black = np.zeros((20, 20), dtype=np.uint8)
    assert empty(white) is True
    assert empty(black) is False
