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
            This function takes the sudoku grid, identifies the digits in the image and returns a numpy matrix of the predicted sudoku puzzle.
        '''
        posx = grid.shape[1] // 9
        posy = grid.shape[0] // 9
        border = config.DIGIT_CROP_BORDER
        digit_size = config.DIGIT_SIZE
        sudoku = np.zeros((9,9), dtype=np.uint8)

        cell_positions = []
        cell_batch = []

        #traverse through each part of the 9x9 puzzle and collect the non-empty cells
        for i in range(9):
            for j in range(9):
                # extract the digit at the particular location
                digit = grid[posy*i : posy*(i+1), posx*j : posx*(j+1)]

                # to check if the block is empty or conatins a digit, extract the center of the image and perform the empty function on it.
                #   if the block contains a digit the center of the iamge will have black pixels
                #   if the block is blank then the center of the image will be mostly white pixels.
                thresholdY = int(0.25 * digit.shape[0])
                thresholdX = int(0.25 * digit.shape[1])
                center = digit[thresholdY: digit.shape[0]-thresholdY, thresholdX: digit.shape[1]-thresholdX]
                if empty(center):
                    # if the block is empty skip (do nothing)
                    continue
                else:
                    # if block contains digit, remove border pixels
                    crop_image = remove_border(digit)
                    #reisize the image to the input size of prediction model - few border pixels
                    resize = cv2.resize(crop_image, (digit_size-2*border, digit_size-2*border), interpolation=cv2.INTER_AREA)
                    #we pad the image with white border as the images used in the model training have some white border pixels
                    padded_digit = cv2.copyMakeBorder(resize, border, border, border, border, cv2.BORDER_CONSTANT, value=(255,255,255))
                    padded_digit = padded_digit.astype('float32')
                    padded_digit = padded_digit/255.0
                    cell_positions.append((i, j))
                    cell_batch.append(padded_digit.reshape(digit_size, digit_size, 1))

        if cell_batch:
            batch = np.stack(cell_batch, axis=0)
            # the model contains 9 classes which start from 0 to 8. The digits in sudoku however range from 1-9.
            # So we add 1 to the prediciton to get the correct number.
            preds = self.model(batch, training=False).numpy().argmax(axis=1) + 1
            for (i, j), pred in zip(cell_positions, preds):
                sudoku[i][j] = pred

        return sudoku
