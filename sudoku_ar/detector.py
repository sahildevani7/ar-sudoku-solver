import cv2
import numpy as np

from . import config


def preprocess(img):
    '''
        This funciton perfroms basic image preprocessing to make it easy to find contours
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    adaptThresh_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    opening = cv2.morphologyEx(adaptThresh_inv, cv2.MORPH_OPEN, kernel)
    return opening


def find_largest_contour(image):
    '''
        The sudoku box will have the largest contour area. This function checks for the
        largest contour that approximates to a quadrilateral (4 vertices after approxPolyDP)
        and returns that simplified 4-point polygon - any other shape (a hand, a stray dark
        region, text on a page) is skipped even if its area is bigger.
    '''
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    biggest = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > config.CONTOUR_MIN_AREA and area > max_area:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                max_area = area
                biggest = approx
    return biggest


def get_corners(biggest_contour):
    '''
        This funciton returns the 4 corner coordinates of the grid.
        In opencv the origin starts at the top left corner of the image,
        so the x axis INCREASES to the RIGHT and DECREASES to the LEFT,
            and y aixis INCREASES downwards and DECREASES upwards

        Hence, the sum of topleft coordinate will have least magnitude and  the sum of bottomright coordinate will have highest magnitude.
        Also the differnece of topright coordinate will have least magnitude and the difference of the bottomleft coordinate will have
            the highest magnitude.

        We return a numpy list of coordinates in the order - [topleft, topright, bottomright, bottomleft]
        THE ORDER IS IMPORTANT
    '''
    coords = np.zeros((4,2), np.float32)
    sumation = biggest_contour.sum(axis=2)
    coords[0] = biggest_contour[np.argmin(sumation)][0]     #topleft
    coords[2] = biggest_contour[np.argmax(sumation)][0]     #bottomright

    difference = np.diff(biggest_contour, axis=2)
    coords[1] = biggest_contour[np.argmin(difference)][0]   #topright
    coords[3] = biggest_contour[np.argmax(difference)][0]   #bottomleft

    return coords


def validate_rect(coords):
    '''
        This function checks if the 4 coordinates form a rectanlge (almost) or not.
            The sudoku grid will be a quadrilateral with almost equal opposite sides

        The points will not form a perfect rectangle so we check if the length of oppposite sides are almost equal
            i.e. the length of smaller side is at least greater than 80% length of the larger side.

        If the condition fails then the points do not form a sudoku grid else the points form a sudoku grid.
    '''
    tleft, tright, bright, bleft = coords

    # using distance formula to calculate the width and height from the 4 coordinates
    widthTop = np.sqrt( ((tright[0] - tleft[0])**2) + ((tright[1] - tleft[1])**2) )
    widthBot = np.sqrt( ((bright[0] - bleft[0])**2) + ((bright[1] - bleft[1])**2) )

    heightRight = np.sqrt(((tright[0] - bright[0]) ** 2) + ((tright[1] - bright[1]) ** 2))
    heightLeft = np.sqrt(((tleft[0] - bleft[0]) ** 2) + ((tleft[1] - bleft[1]) ** 2))

    # the differnce between the lengths of opposited sides must be less than the tolerance fraction of the length of the larger side
    deltaH = config.RECT_SIDE_TOLERANCE * max(heightLeft, heightRight)
    deltaW = config.RECT_SIDE_TOLERANCE * max(widthBot, widthTop)

    if abs(widthTop-widthBot)<deltaW and abs(heightRight-heightLeft)<deltaH:
        return True
    return False


def _downscale(frame, max_dim):
    h, w = frame.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = frame
        scale = 1.0
    return small, scale


def find_grid(frame):
    '''
        Detects the sudoku grid quad in `frame`. Contour search runs on a copy downscaled to
        config.DETECTION_MAX_DIM (a 1080p frame otherwise costs ~7x a 640px-wide one for the
        same search). Returns (biggest_contour, coords), both scaled back to frame's original
        resolution, or (None, None) if no contour was found.

        Callers should still run validate_rect(coords) themselves - that's a separate check
        for whether the found contour is actually a well-formed quad.
    '''
    small, scale = _downscale(frame, config.DETECTION_MAX_DIM)
    processed = preprocess(small)
    biggest = find_largest_contour(processed)
    if biggest is None:
        return None, None

    if scale != 1.0:
        biggest = np.round(biggest.astype(np.float32) / scale).astype(np.int32)

    coords = get_corners(biggest)
    return biggest, coords


def perspective_transform(coords, image):
    '''
        This funtion returns a birds eye view of the extracted sudoku grid from the frame,
        warped to a fixed WARP_SIZE x WARP_SIZE square so downstream cell slicing and model
        input sizing stay constant regardless of how the grid is framed.
    '''
    # create a destination array with points [topleft, topright, bottomright, bottomleft]
    # The topleft corner is the origin.
    size = config.WARP_SIZE
    dst = np.array([
        [0, 0],
        [size - 1, 0],
        [size - 1, size - 1],
        [0, size - 1]], dtype = "float32" )

    M = cv2.getPerspectiveTransform(coords, dst)
    warped = cv2.warpPerspective(image, M, (size, size))

    return warped
