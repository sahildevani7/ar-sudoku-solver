MODEL_PATH = "models/sudoku_digit_recognizer.h5"
CAMERA_INDEX = 0

# Grid detection
CONTOUR_MIN_AREA = 1000
RECT_SIDE_TOLERANCE = 0.2  # opposite sides of the detected quad may differ by up to this fraction
DETECTION_MAX_DIM = 640    # contour search runs on a copy downscaled to this long side, then scales back
CORNER_SMOOTHING = 0.4     # EMA weight given to each new corner reading (lower = smoother, more lag)

# Shape gates separating a real grid outline from any other large dark region (a face, a hand,
# a shadow). validate_rect alone only compares opposite side lengths, which an irregular blob's
# extremal points pass easily - these are what actually reject it.
#   solidity = contour area / area of the 4-corner quad. A true rectangle traces its own quad
#   (~1.0); a rounded/irregular blob's contour and extremal-point quad disagree sharply.
QUAD_SOLIDITY_MIN = 0.80
QUAD_SOLIDITY_MAX = 1.25
#   a sudoku grid is square; allow generous perspective skew but not a tall/wide silhouette
QUAD_ASPECT_MIN = 0.55
QUAD_ASPECT_MAX = 1.80

# Perspective warp - fixed square so cell slicing and model input sizing stay
# constant frame-to-frame regardless of how the grid is framed by the camera.
WARP_SIZE = 450  # divisible by 9 -> exact 50px cells

# Digit recognition
DIGIT_SIZE = 32
DIGIT_CROP_BORDER = 3
GRID_BORDER_CROP_RATIO = 0.12
EMPTY_CELL_WHITE_RATIO = 0.97
MIN_CONFIDENCE = 0.75  # reject a whole read if any predicted cell falls below this softmax confidence
# Measured on real camera captures: a well-centred read scores ~0.91 at worst, while frames of
# noise/glare score 0.2-0.5, so this sits between. (An earlier 0.85 was guesswork and rejected
# perfectly good reads.)

# Each cell's digit is isolated, cropped to its own bounding box and re-centred before going to
# the model, which was trained on centred digits. Feeding it whatever happened to land in the
# cell crop instead cost both accuracy and confidence.
DIGIT_TARGET_SIZE = 20        # longest side of the digit, within the DIGIT_SIZE canvas
DIGIT_MIN_AREA_RATIO = 0.008  # ink blobs smaller than this fraction of a cell are speckle
DIGIT_MIN_HEIGHT_RATIO = 0.25 # ...and a real digit is at least this tall relative to the cell

# Printed digits within one puzzle are all about the same height, so anything markedly shorter
# than the typical candidate is not a digit - on a phone screen it is usually a streak of glare.
# Measured on a real capture: genuine digits filled 0.49-0.60 of a cell, glare streaks 0.45-0.46.
# Judging against this grid's own median rather than a fixed ratio keeps it working across
# different digit sizes and fonts. The margin here is narrow and tuned on limited data; if real
# digits start disappearing, lower it.
DIGIT_MIN_RELATIVE_HEIGHT = 0.88

# Grid line positions are fitted to find the true cell boundaries. The detected quad rarely lands
# exactly on the puzzle's edges, and assuming nine even slices then drifts by a whole row partway
# down the grid. Fitting the ruling instead tolerates a warp that overshoots or clips the border.
CELL_FIT_MIN_LINES = 6        # fewer detected lines than this and the fit isn't trustworthy
GRID_LINE_PROFILE_THRESHOLD = 0.35  # fraction of the strongest profile peak counted as a line
# The grid's own ruled lines land inside the cell crops whenever the warp is even slightly off,
# which makes empty cells look occupied and feeds line fragments to the model. Structures at
# least this fraction of the grid's width/height are treated as ruling and erased before cells
# are cut - a digit is never that long in one direction.
GRID_LINE_LENGTH_RATIO = 1 / 15

# Acquisition state machine
CONFIRM_FRAMES = 3       # consecutive identical reads required before locking onto a puzzle
LOST_FRAMES = 15         # frames the grid may be briefly absent/occluded before releasing lock
REJECT_COOLDOWN_FRAMES = 45  # frames to ignore a read that already failed to solve, so it isn't retried every frame
MIN_GIVENS = 17          # the minimum clues a uniquely-solvable sudoku can have; fewer implies a bad read
                         # (this solver's runtime is highly sensitive to sparse/near-empty grids, so this
                         # also guards against handing it a pathological case that could hang for a long time)
