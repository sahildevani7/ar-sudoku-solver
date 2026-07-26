import os
import sys
import time
from enum import Enum, auto

import cv2
import numpy as np

from . import config, detector, hud, overlay
from . import recognizer as recognizer_module
from .capture import Camera
from .recognizer import DigitRecognizer
from .solver import solve_wrapper
from .validator import isValidConfig

GREEN = (0, 255, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)

DEBUG_DIR = "/tmp/sudoku_debug"
DEBUG_DUMP_EVERY = 6      # frames between scene snapshots - dumping every frame costs more than detection
DEBUG_SCENE_SLOTS = 12    # rotating snapshot files, so earlier frames survive until quit


def _flat(grid):
    return "".join(str(v) for v in grid.flatten())


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

    def __init__(self, recognizer, debug=False):
        self.recognizer = recognizer
        self.debug = debug
        self.debug_frame_count = 0
        self.debug_slot = 0
        self.reset()

    def _log(self, msg):
        if self.debug:
            print(msg, file=sys.stderr)

    def _dump_debug_frames(self, warped, warped_inv):
        # overwrites the same files each time - a live snapshot of what recognition is currently
        # looking at. These are small (WARP_SIZE square), so they're written lossless: JPEG
        # artifacts on a binarised image destroy exactly the thin-line detail worth inspecting.
        os.makedirs(DEBUG_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(DEBUG_DIR, "warped.png"), warped)
        cv2.imwrite(os.path.join(DEBUG_DIR, "warped_inv.png"), warped_inv)
        cv2.imwrite(os.path.join(DEBUG_DIR, "digits_only.png"), recognizer_module.remove_grid_lines(warped_inv))

    def _dump_debug_scene(self, frame, quad, coords, is_grid):
        '''
            Full-frame snapshots with the chosen contour drawn on.

            Writes to a rotating set of numbered slots rather than one file: overwriting a single
            path only ever preserves the final frame before quit, which is usually *after* the
            puzzle has been lowered out of shot - useless for diagnosing what happened while it
            was actually held up. Also throttled and JPEG-encoded, since doing this every frame
            at full resolution costs more than the whole detection pipeline.
        '''
        self.debug_frame_count += 1
        if self.debug_frame_count % DEBUG_DUMP_EVERY:
            return

        os.makedirs(DEBUG_DIR, exist_ok=True)
        scene = frame.copy()
        if quad is not None:
            color = GREEN if is_grid else RED
            cv2.drawContours(scene, [quad], 0, color, 3)
            label = "is_grid=%s solidity=%.2f aspect=%.2f" % (
                is_grid, detector.quad_solidity(quad, coords), detector.quad_aspect(coords))
        else:
            color, label = RED, "no contour above area threshold"
        cv2.putText(scene, label, (10, scene.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imwrite(os.path.join(DEBUG_DIR, "scene_%02d.jpg" % self.debug_slot), scene)
        self.debug_slot = (self.debug_slot + 1) % DEBUG_SCENE_SLOTS
        # a separate always-latest copy of any frame that actually passed the shape gates
        if is_grid:
            cv2.imwrite(os.path.join(DEBUG_DIR, "accepted.jpg"), scene)

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

        biggest, coords, is_grid = detector.find_grid(frame)

        if self.debug:
            self._dump_debug_scene(frame, biggest, coords, is_grid)

        if biggest is None:
            self._on_grid_absent()
            hud.draw_status(frame, "No grid detected", RED)
            return frame

        if not is_grid:
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

        # the detected quad rarely lands exactly on the puzzle's border, so cut cells at the
        # ruling actually found rather than at nine even slices
        boundaries = detector.find_cell_boundaries(warped_inv)
        digits, confident, min_confidence = self.recognizer.extract_digit(warped_inv, boundaries)
        givens = np.count_nonzero(digits)

        if self.debug and self.debug_frame_count % DEBUG_DUMP_EVERY == 0:
            self._dump_debug_frames(warped, warped_inv)

        if not confident:
            self._log("[low-confidence] min_conf=%.2f givens=%d grid=%s" % (min_confidence, givens, _flat(digits)))
            self.candidate = None
            self.confirm_count = 0
            self.state = State.SEARCHING
            hud.draw_status(frame, "Hold steady - low confidence read", ORANGE)
            return frame

        # isValidConfig alone would accept a nearly-empty misread; MIN_GIVENS also guards
        # against handing the solver a pathologically sparse grid (see config.py).
        valid_config = isValidConfig(digits)
        if not (valid_config and givens >= config.MIN_GIVENS):
            self._log("[rejected-read] givens=%d (need >=%d) valid_config=%s grid=%s" %
                      (givens, config.MIN_GIVENS, valid_config, _flat(digits)))
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
            if self.candidate is not None:
                self._log("[mismatch] previous=%s new=%s" % (_flat(self.candidate), _flat(digits)))
            self.candidate = digits.copy()
            self.confirm_count = 1
        self.state = State.CONFIRMING

        if self.confirm_count < config.CONFIRM_FRAMES:
            hud.draw_status(frame, "Confirming... (%d/%d)" % (self.confirm_count, config.CONFIRM_FRAMES), GREEN)
            return frame

        solved, _ = solve_wrapper(self.candidate.copy())
        if solved is not None:
            self._log("[solved] grid=%s" % _flat(self.candidate))
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
            self._log("[unsolvable] grid=%s" % _flat(self.candidate))
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


def run(source=None, debug=False):
    recognizer = DigitRecognizer()
    pipeline = SudokuPipeline(recognizer, debug=debug)

    if source is None:
        # Camera's background-thread "always serve the newest frame" behavior only makes sense
        # against a live camera outrunning processing speed - a video file already plays back
        # deterministically frame-by-frame, and racing a reader thread through it would just
        # skip almost the whole file before the main loop got to see any of it.
        cap = Camera(config.CAMERA_INDEX).start()
    else:
        cap = cv2.VideoCapture(source)

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
