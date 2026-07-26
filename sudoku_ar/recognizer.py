import cv2
import numpy as np
import tensorflow as tf

from . import config


def remove_border(binary_image):
    '''
    This function removes the boundary pixels of the image
    '''
    x = binary_image.shape[1]
    y = binary_image.shape[0]
    border = int(config.GRID_BORDER_CROP_RATIO * x)
    roi = binary_image[border:y-border, border:x-border]
    return roi


def remove_grid_lines(image):
    '''
        Erases the puzzle's ruled lines from a binarised grid (black ink on white), leaving only
        the digits.

        Without this, any small warp error puts ruling inside a cell's centre crop, so empty()
        reports nearly every cell as occupied and the model is handed line fragments to
        classify. Isolating runs at least config.GRID_LINE_LENGTH_RATIO of the image's
        width/height picks out ruling specifically - no digit is that long in one direction.
    '''
    ink = (image < 128).astype(np.uint8) * 255
    height, width = ink.shape[:2]

    h_len = max(3, int(width * config.GRID_LINE_LENGTH_RATIO))
    v_len = max(3, int(height * config.GRID_LINE_LENGTH_RATIO))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))

    lines = cv2.bitwise_or(horizontal, vertical)
    # widen slightly so anti-aliased edges of each line go with it
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8), iterations=1)

    digits_only = cv2.bitwise_and(ink, cv2.bitwise_not(lines))
    return cv2.bitwise_not(digits_only)  # back to black ink on white


def extract_centered_digit(cell):
    '''
        Isolates the digit in a single cell crop (black ink on white) and returns it centred on a
        DIGIT_SIZE canvas, ready for the model - or None if the cell holds no digit.

        The model was trained on centred digits, so handing it the raw cell crop (with the digit
        wherever it happened to fall, at whatever scale) costs both accuracy and confidence. On a
        real capture, centring corrected two of three misread digits and lifted the worst-cell
        confidence from 0.72 to 0.91. The largest qualifying ink blob is taken as the digit,
        which also discards speckle and leftover line fragments.
    '''
    ink = (cell < 128).astype(np.uint8)
    height, width = ink.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)

    min_area = config.DIGIT_MIN_AREA_RATIO * height * width
    min_height = config.DIGIT_MIN_HEIGHT_RATIO * height
    best = None
    for index in range(1, count):  # 0 is the background label
        _, _, _, blob_h, area = stats[index]
        if area < min_area or blob_h < min_height:
            continue
        if best is None or area > stats[best][4]:
            best = index
    if best is None:
        return None

    x, y, w, h, _ = stats[best]
    digit = (labels[y:y + h, x:x + w] == best).astype(np.uint8) * 255

    scale = config.DIGIT_TARGET_SIZE / max(w, h)
    digit = cv2.resize(digit, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((config.DIGIT_SIZE, config.DIGIT_SIZE), dtype=np.uint8)
    offset_y = (config.DIGIT_SIZE - digit.shape[0]) // 2
    offset_x = (config.DIGIT_SIZE - digit.shape[1]) // 2
    canvas[offset_y:offset_y + digit.shape[0], offset_x:offset_x + digit.shape[1]] = digit

    # canvas is white ink on black; the model expects dark digits on a white field
    return 1.0 - (canvas.astype('float32') / 255.0)


def empty(image):
    '''
        The digits are written in black on white background.
        If only less than 3% of the pixels are black, we can safely assume that the image is empty (only contains some noise),
            else the image contains a digit.
        The countNonZero function returns the number of non black pixels, that in our case is white pixels. So it is returning
            the number of white pixels. If white pixels contain more than 97%(100-3) of the image, we declare it to be empty
    '''
    if cv2.countNonZero(image) >= config.EMPTY_CELL_WHITE_RATIO*(image.shape[0] * image.shape[1]):
        return True
    else:
        return False


class DigitRecognizer:
    '''
    Wraps the digit-recognition CNN. All non-empty cells of a grid are batched into a single
    model call instead of predicting one cell at a time - Keras incurs large fixed per-call
    overhead, so 81 individual predict() calls cost ~1.6s while one batched call over the same
    81 cells costs ~15ms.
    '''

    def __init__(self, model_path=config.MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)
        # warm the model once at startup so the first real frame doesn't pay tf.function trace cost
        self.model(np.zeros((1, config.DIGIT_SIZE, config.DIGIT_SIZE, 1), dtype=np.float32), training=False)

    def extract_digit(self, grid):
        '''
            This function takes the sudoku grid, identifies the digits in the image and returns:
              - a numpy matrix of the predicted sudoku puzzle
              - a bool that is False if any predicted cell's softmax confidence fell below
                config.MIN_CONFIDENCE, signalling the whole read should be treated as unreliable
              - the lowest per-cell softmax confidence seen among non-empty cells (1.0 if none),
                for diagnostics
        '''
        grid = remove_grid_lines(grid)

        posx = grid.shape[1] // 9
        posy = grid.shape[0] // 9
        digit_size = config.DIGIT_SIZE
        sudoku = np.zeros((9,9), dtype=np.uint8)

        cell_positions = []
        cell_batch = []

        #traverse through each part of the 9x9 puzzle and collect the cells that hold a digit
        for i in range(9):
            for j in range(9):
                cell = grid[posy*i : posy*(i+1), posx*j : posx*(j+1)]
                prepared = extract_centered_digit(cell)
                if prepared is None:
                    # no qualifying ink in this cell - it's blank
                    continue
                cell_positions.append((i, j))
                cell_batch.append(prepared.reshape(digit_size, digit_size, 1))

        confident = True
        min_confidence = 1.0
        if cell_batch:
            batch = np.stack(cell_batch, axis=0)
            probs = self.model(batch, training=False).numpy()
            cell_confidences = probs.max(axis=1)
            min_confidence = float(cell_confidences.min())
            confident = bool(min_confidence >= config.MIN_CONFIDENCE)
            # the model contains 9 classes which start from 0 to 8. The digits in sudoku however range from 1-9.
            # So we add 1 to the prediciton to get the correct number.
            preds = probs.argmax(axis=1) + 1
            for (i, j), pred in zip(cell_positions, preds):
                sudoku[i][j] = pred

        return sudoku, confident, min_confidence
