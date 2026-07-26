import time
from enum import Enum, auto

import cv2
import numpy as np

from . import config, detector, hud, overlay
from .capture import Camera
from .recognizer import DigitRecognizer
from .solver import solve_wrapper
from .validator import isValidConfig

GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)


class State(Enum):
    SEARCHING = auto()   # no candidate grid yet
    CONFIRMING = auto()  # a candidate is being read repeatedly, waiting for it to stabilize
    SOLVED = auto()      # locked onto a solved puzzle; recognition is skipped entirely


class SudokuPipeline:
    '''
    Tracks acquisition of a sudoku puzzle across frames.

    SEARCHING/CONFIRMING require the same digit read for config.CONFIRM_FRAMES consecutive
    frames before it's trusted enough to solve - this kills transient misreads that would
    otherwise flicker into a wrong solution. Once SOLVED, recognition stops entirely (only
    corner tracking + overlay run per frame) until the grid is absent for config.LOST_FRAMES
    frames, or the user presses 'r', at which point it releases back to SEARCHING so a new
    puzzle can be picked up.
    '''

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.reset()

    def reset(self):
        self.state = State.SEARCHING
        self.candidate = None
        self.confirm_count = 0
        self.lost_count = 0
        self.rejected_grid = None
        self.rejected_cooldown = 0
        self.solved_grid = None
        self.unsolved_grid = None
        self.canvas = None
        self.mask = None
        self.smoothed_coords = None

    def process(self, frame):
        if self.rejected_cooldown > 0:
            self.rejected_cooldown -= 1

        biggest, coords = detector.find_grid(frame)

        if biggest is None:
            self._on_grid_absent()
            hud.draw_status(frame, "No grid detected", RED)
            return frame

        if not detector.validate_rect(coords):
            self._on_grid_absent()
            hud.draw_status(frame, "Adjust grid position for better visibility", RED)
            return frame

        # a valid quad is in view this frame
        self.lost_count = 0
        coords = self._smooth(coords)
        cv2.drawContours(frame, [biggest], 0, GREEN, 2)

        if self.state == State.SOLVED:
            frame = overlay.composite(frame, self.canvas, self.mask, coords, config.WARP_SIZE)
            hud.draw_status(frame, "Solved! Press 'r' to reset", GREEN)
            return frame

        warped = detector.perspective_transform(coords, frame)
        return self._acquire(frame, warped)

    def _smooth(self, coords):
        if self.smoothed_coords is None:
            self.smoothed_coords = coords
        else:
            alpha = config.CORNER_SMOOTHING
            self.smoothed_coords = alpha * coords + (1 - alpha) * self.smoothed_coords
        return self.smoothed_coords

    def _acquire(self, frame, warped):
        warped_binary = detector.preprocess(warped)
        warped_inv = cv2.bitwise_not(warped_binary)
        digits, confident = self.recognizer.extract_digit(warped_inv)

        if not confident:
            self.candidate = None
            self.confirm_count = 0
            self.state = State.SEARCHING
            hud.draw_status(frame, "Hold steady - low confidence read", ORANGE)
            return frame

        # isValidConfig alone would accept a nearly-empty misread; MIN_GIVENS also guards
        # against handing the solver a pathologically sparse grid (see config.py).
        if not (isValidConfig(digits) and np.count_nonzero(digits) >= config.MIN_GIVENS):
            self.candidate = None
            self.confirm_count = 0
            self.state = State.SEARCHING
            hud.draw_status(frame, "Processing grid...", GREEN)
            return frame

        if self.rejected_cooldown > 0 and self.rejected_grid is not None and np.array_equal(digits, self.rejected_grid):
            hud.draw_status(frame, "Misread detected - re-scanning...", ORANGE)
            return frame

        if self.candidate is not None and np.array_equal(digits, self.candidate):
            self.confirm_count += 1
        else:
            self.candidate = digits.copy()
            self.confirm_count = 1
        self.state = State.CONFIRMING

        if self.confirm_count < config.CONFIRM_FRAMES:
            hud.draw_status(frame, "Confirming... (%d/%d)" % (self.confirm_count, config.CONFIRM_FRAMES), GREEN)
            return frame

        solved, _ = solve_wrapper(self.candidate.copy())
        if solved is not None:
            self.solved_grid = solved
            self.unsolved_grid = self.candidate.copy()
            self.canvas, self.mask = overlay.render_solution_canvas(solved, self.unsolved_grid, config.WARP_SIZE)
            self.state = State.SOLVED
            self.candidate = None
            self.confirm_count = 0
            self.rejected_grid = None
            self.rejected_cooldown = 0
            hud.draw_status(frame, "Solved! Press 'r' to reset", GREEN)
        else:
            # remember this exact misread so it isn't retried every single frame
            self.rejected_grid = self.candidate.copy()
            self.rejected_cooldown = config.REJECT_COOLDOWN_FRAMES
            self.candidate = None
            self.confirm_count = 0
            self.state = State.SEARCHING
            hud.draw_status(frame, "Misread detected - re-scanning...", ORANGE)

        return frame

    def _on_grid_absent(self):
        self.lost_count += 1
        if self.lost_count >= config.LOST_FRAMES:
            self.reset()


def run():
    recognizer = DigitRecognizer()
    pipeline = SudokuPipeline(recognizer)

    cap = Camera(config.CAMERA_INDEX).start()

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        hud.draw_instructions(frame)
        frame = pipeline.process(frame)

        elapsed = time.time() - start_time
        fps = int(1 / elapsed) if elapsed > 0 else 0
        hud.draw_fps(frame, fps)

        cv2.imshow('Sudoku Solver', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            pipeline.reset()

    cap.release()
    cv2.destroyAllWindows()
