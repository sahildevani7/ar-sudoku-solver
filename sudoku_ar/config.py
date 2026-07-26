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
