from unittest import mock

import numpy as np

from sudoku_ar import config
from sudoku_ar.pipeline import SudokuPipeline, State

from tests.test_solver import PUZZLE, SOLUTION

DUMMY_CONTOUR = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
DUMMY_COORDS = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)


def make_frame():
    return np.zeros((60, 60, 3), dtype=np.uint8)


class StubRecognizer:
    '''Returns a fixed (grid, confident, min_confidence) triple on every call, ignoring the input image.'''

    def __init__(self, grid, confident=True):
        self.grid = grid
        self.confident = confident
        self.calls = 0

    def extract_digit(self, warped_inv):
        self.calls += 1
        return self.grid.copy(), self.confident, (1.0 if self.confident else 0.0)


def grid_present(monkeypatch):
    '''Patches the detector boundary so pipeline.process() believes a valid grid quad is in view.'''
    monkeypatch.setattr("sudoku_ar.pipeline.detector.preprocess",
                        lambda img: np.zeros((config.WARP_SIZE, config.WARP_SIZE), dtype=np.uint8))
    monkeypatch.setattr("sudoku_ar.pipeline.detector.find_grid", lambda frame: (DUMMY_CONTOUR, DUMMY_COORDS, True))
    monkeypatch.setattr("sudoku_ar.pipeline.detector.validate_rect", lambda coords: True)
    monkeypatch.setattr("sudoku_ar.pipeline.detector.perspective_transform",
                         lambda coords, img: np.zeros((config.WARP_SIZE, config.WARP_SIZE, 3), dtype=np.uint8))


def grid_absent(monkeypatch):
    monkeypatch.setattr("sudoku_ar.pipeline.detector.preprocess",
                        lambda img: np.zeros((config.WARP_SIZE, config.WARP_SIZE), dtype=np.uint8))
    monkeypatch.setattr("sudoku_ar.pipeline.detector.find_grid", lambda frame: (None, None, False))


def test_confirm_then_lock_solves_after_matching_frames(monkeypatch):
    grid_present(monkeypatch)
    pipeline = SudokuPipeline(StubRecognizer(PUZZLE))

    for _ in range(config.CONFIRM_FRAMES - 1):
        pipeline.process(make_frame())
        assert pipeline.state == State.CONFIRMING

    pipeline.process(make_frame())
    assert pipeline.state == State.SOLVED
    assert np.array_equal(pipeline.solved_grid, SOLUTION)
    assert np.array_equal(pipeline.unsolved_grid, PUZZLE)


def test_mismatched_reads_never_lock(monkeypatch):
    grid_present(monkeypatch)
    other_grid = PUZZLE.copy()
    other_grid[8, 8] = 9 if other_grid[8, 8] != 9 else 8  # a different, still-valid-looking read

    recognizer = StubRecognizer(PUZZLE)
    pipeline = SudokuPipeline(recognizer)

    for i in range(10):
        recognizer.grid = PUZZLE if i % 2 == 0 else other_grid
        pipeline.process(make_frame())
        assert pipeline.state != State.SOLVED
        assert pipeline.confirm_count <= 1


def test_low_confidence_read_is_rejected(monkeypatch):
    grid_present(monkeypatch)
    pipeline = SudokuPipeline(StubRecognizer(PUZZLE, confident=False))

    for _ in range(config.CONFIRM_FRAMES + 2):
        pipeline.process(make_frame())
        assert pipeline.state == State.SEARCHING
        assert pipeline.candidate is None


def test_min_givens_guard_blocks_sparse_reads(monkeypatch):
    grid_present(monkeypatch)
    sparse = np.zeros((9, 9), dtype=np.uint8)
    sparse[0, 0] = 4
    sparse[4, 4] = 5
    assert np.count_nonzero(sparse) < config.MIN_GIVENS

    with mock.patch("sudoku_ar.pipeline.solve_wrapper") as solve_mock:
        pipeline = SudokuPipeline(StubRecognizer(sparse))
        for _ in range(config.CONFIRM_FRAMES + 2):
            pipeline.process(make_frame())
        assert pipeline.state == State.SEARCHING
        solve_mock.assert_not_called()


def test_unsolvable_grid_is_remembered_and_not_retried(monkeypatch):
    grid_present(monkeypatch)
    # Enough givens to pass the MIN_GIVENS/isValidConfig gate, but solve_wrapper is forced to fail.
    grid = np.where(np.arange(81).reshape(9, 9) % 3 == 0, SOLUTION, 0).astype(np.uint8)
    assert np.count_nonzero(grid) >= config.MIN_GIVENS

    with mock.patch("sudoku_ar.pipeline.solve_wrapper", return_value=(None, None)) as solve_mock:
        pipeline = SudokuPipeline(StubRecognizer(grid))

        for _ in range(config.CONFIRM_FRAMES):
            pipeline.process(make_frame())
        assert pipeline.state == State.SEARCHING
        assert pipeline.rejected_grid is not None
        assert solve_mock.call_count == 1

        # Same misread fed again immediately - should be ignored during cooldown, not re-solved.
        for _ in range(5):
            pipeline.process(make_frame())
        assert solve_mock.call_count == 1
        assert pipeline.state == State.SEARCHING


def test_brief_occlusion_does_not_release_lock(monkeypatch):
    grid_present(monkeypatch)
    pipeline = SudokuPipeline(StubRecognizer(PUZZLE))
    for _ in range(config.CONFIRM_FRAMES):
        pipeline.process(make_frame())
    assert pipeline.state == State.SOLVED

    grid_absent(monkeypatch)
    for _ in range(config.LOST_FRAMES - 1):
        pipeline.process(make_frame())
    assert pipeline.state == State.SOLVED  # not yet released

    grid_present(monkeypatch)
    pipeline.process(make_frame())
    assert pipeline.state == State.SOLVED
    assert np.array_equal(pipeline.solved_grid, SOLUTION)


def test_sustained_absence_releases_lock(monkeypatch):
    grid_present(monkeypatch)
    pipeline = SudokuPipeline(StubRecognizer(PUZZLE))
    for _ in range(config.CONFIRM_FRAMES):
        pipeline.process(make_frame())
    assert pipeline.state == State.SOLVED

    grid_absent(monkeypatch)
    for _ in range(config.LOST_FRAMES):
        pipeline.process(make_frame())
    assert pipeline.state == State.SEARCHING
    assert pipeline.solved_grid is None


def test_manual_reset_clears_lock(monkeypatch):
    grid_present(monkeypatch)
    pipeline = SudokuPipeline(StubRecognizer(PUZZLE))
    for _ in range(config.CONFIRM_FRAMES):
        pipeline.process(make_frame())
    assert pipeline.state == State.SOLVED

    pipeline.reset()
    assert pipeline.state == State.SEARCHING
    assert pipeline.solved_grid is None
    assert pipeline.candidate is None
