import time

import cv2
import numpy as np

from . import config, detector, hud, overlay
from .recognizer import DigitRecognizer
from .solver import solve_wrapper
from .validator import isValidConfig


def run():
    '''
        This is where the whole procedure takes place.
        The steps are:
            -> preprocess the frame
            -> find largest contour (which is expected to be the sudoku grid box)
            -> find the corners of the largest contour
            -> check if the corners approximately form a rectangle
            -> extract the grid image
            -> divide it into 9x9 blocks and perfrom digit prediction to find sudoku matrix
            -> check if resultant sudoku matrix is valid
            -> solve the sudoku
            -> write the result on the extracted warped image
            -> place this final solved image onto the frame
            -> show the frame.
    '''
    recognizer = DigitRecognizer()

    #initialize an empty 9x9 matrix
    sudoku_matrix = np.zeros((9,9), dtype=np.uint8)
    #set a boolean flag to false
    validation = False

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()

        hud.draw_instructions(frame)

        processedFrame = detector.preprocess(frame)
        biggest = detector.find_largest_contour(processedFrame)

        try:
            coords = detector.get_corners(biggest)
            if detector.validate_rect(coords):
                # Draw green contour around detected grid
                cv2.drawContours(frame, [biggest], 0, (0,255,0), 2)

                # Show processing status
                if not validation:
                    hud.draw_status(frame, "Processing grid...", (0, 255, 0))

                warped = detector.perspective_transform(coords, frame)
                warped_binary = detector.preprocess(warped)
                warped_inv = cv2.bitwise_not(warped_binary)

                if not validation:
                    sudoku_matrix = recognizer.extract_digit(warped_inv)
                    unsolved = sudoku_matrix.copy()
                    if isValidConfig(sudoku_matrix) and np.count_nonzero(sudoku_matrix)!=0:
                        validation = True
                        sudoku_matrix, solve_time = solve_wrapper(sudoku_matrix)
                        # Show success message
                        hud.draw_status(frame, "Grid processed successfully!", (0, 255, 0))

                solved_grid_image = overlay.fill_sudoku(sudoku_matrix, unsolved, warped)
                frame = overlay.unwarp_image(solved_grid_image, frame, coords)
            else:
                # Show message when grid is not properly visible
                hud.draw_status(frame, "Adjust grid position for better visibility", (0, 0, 255))

        except Exception:
            # Show message when no grid is detected
            hud.draw_status(frame, "No grid detected", (0, 0, 255))

        # Calculate and display FPS
        fps = int(1/(time.time() - start_time))
        hud.draw_fps(frame, fps)

        cv2.imshow('Sudoku Solver', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
