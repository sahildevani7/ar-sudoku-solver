import cv2
import numpy as np
import pytest

from sudoku_ar import config
from sudoku_ar.recognizer import DigitRecognizer, remove_border, empty


@pytest.fixture(scope="module")
def recognizer():
    return DigitRecognizer()


def blank_grid():
    return np.full((config.WARP_SIZE, config.WARP_SIZE), 255, dtype=np.uint8)


def test_blank_grid_returns_all_zero_and_confident(recognizer):
    result, confident = recognizer.extract_digit(blank_grid())
    assert result.shape == (9, 9)
    assert result.dtype == np.uint8
    assert np.all(result == 0)
    assert confident is True


def test_non_empty_cells_produce_in_range_digits(recognizer):
    grid = blank_grid()
    posx = posy = config.WARP_SIZE // 9
    # draw dark blobs in a few cells so empty() reports them as non-empty
    for (i, j) in [(2, 3), (5, 7), (0, 0)]:
        cx, cy = j * posx + posx // 2, i * posy + posy // 2
        cv2.circle(grid, (cx, cy), posx // 4, 0, -1)

    result, confident = recognizer.extract_digit(grid)
    assert result.shape == (9, 9)
    assert isinstance(confident, bool)
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

    result, _ = recognizer.extract_digit(grid)

    border = config.DIGIT_CROP_BORDER
    digit_size = config.DIGIT_SIZE
    cell = grid[posy * target_i: posy * (target_i + 1), posx * target_j: posx * (target_j + 1)]
    cropped = remove_border(cell)
    resized = cv2.resize(cropped, (digit_size - 2 * border, digit_size - 2 * border), interpolation=cv2.INTER_AREA)
    padded = cv2.copyMakeBorder(resized, border, border, border, border, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    padded = padded.astype('float32') / 255.0

    direct_pred = recognizer.model(padded.reshape(1, digit_size, digit_size, 1), training=False).numpy().argmax(axis=1)[0] + 1
    assert result[target_i, target_j] == direct_pred


def test_empty_helper_on_mostly_white_and_mostly_black():
    white = np.full((20, 20), 255, dtype=np.uint8)
    black = np.zeros((20, 20), dtype=np.uint8)
    assert empty(white) is True
    assert empty(black) is False
