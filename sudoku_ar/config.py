MODEL_PATH = "models/sudoku_digit_recognizer.h5"
CAMERA_INDEX = 0

# Grid detection
CONTOUR_MIN_AREA = 1000
RECT_SIDE_TOLERANCE = 0.2  # opposite sides of the detected quad may differ by up to this fraction

# Perspective warp - fixed square so cell slicing and model input sizing stay
# constant frame-to-frame regardless of how the grid is framed by the camera.
WARP_SIZE = 450  # divisible by 9 -> exact 50px cells

# Digit recognition
DIGIT_SIZE = 32
DIGIT_CROP_BORDER = 3
GRID_BORDER_CROP_RATIO = 0.12
EMPTY_CELL_WHITE_RATIO = 0.97
MIN_CONFIDENCE = 0.85  # reject a whole read if any predicted cell falls below this softmax confidence

# Acquisition state machine
CONFIRM_FRAMES = 3       # consecutive identical reads required before locking onto a puzzle
LOST_FRAMES = 15         # frames the grid may be briefly absent/occluded before releasing lock
REJECT_COOLDOWN_FRAMES = 45  # frames to ignore a read that already failed to solve, so it isn't retried every frame
MIN_GIVENS = 17          # the minimum clues a uniquely-solvable sudoku can have; fewer implies a bad read
                         # (this solver's runtime is highly sensitive to sparse/near-empty grids, so this
                         # also guards against handing it a pathological case that could hang for a long time)
