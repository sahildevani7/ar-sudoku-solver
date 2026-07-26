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


QUAD_EPSILON_FACTORS = (0.02, 0.03, 0.04, 0.05)
MAX_CONTOUR_CANDIDATES = 8


def _simplify_to_quad(contour):
    '''
        Simplifies a contour to a clean 4-point polygon via approxPolyDP, tried at a few
        increasing epsilons (real camera contours - lighting, blur, compression noise - often
        don't converge to exactly 4 points at a single fixed epsilon the way a clean synthetic
        shape does).

        Returns None if none of them converge to 4 points. It deliberately does NOT fall back to
        the raw contour: get_corners would then force any blob into a "quad" via its extremal
        points, which is exactly how a face silhouette used to get accepted as a grid.
    '''
    perimeter = cv2.arcLength(contour, True)
    for eps_factor in QUAD_EPSILON_FACTORS:
        approx = cv2.approxPolyDP(contour, eps_factor * perimeter, True)
        if len(approx) == 4:
            return approx
    return None


def quad_solidity(contour, coords):
    '''
        Ratio of the contour's own area to the area of the 4-corner quad through it. A real
        rectangular outline traces its own quad (~1.0); a rounded or irregular silhouette
        disagrees with its extremal-point quad substantially in either direction.
    '''
    quad_area = cv2.contourArea(coords.astype(np.float32).reshape(-1, 1, 2))
    if quad_area <= 0:
        return 0.0
    return cv2.contourArea(contour) / quad_area


def quad_aspect(coords):
    '''Width/height ratio of the quad, averaging each pair of opposite sides.'''
    tleft, tright, bright, bleft = coords
    width = (np.linalg.norm(tright - tleft) + np.linalg.norm(bright - bleft)) / 2
    height = (np.linalg.norm(bleft - tleft) + np.linalg.norm(bright - tright)) / 2
    if height <= 0:
        return 0.0
    return width / height


def is_grid_quad(contour, coords):
    '''
        Full shape test for "is this contour plausibly a sudoku grid outline". validate_rect
        alone is too weak - it only compares opposite side lengths, which an irregular blob's
        extremal points pass easily - so this also requires the contour to actually fill its
        quad (solidity) and to be roughly square (aspect).
    '''
    if not validate_rect(coords):
        return False
    solidity = quad_solidity(contour, coords)
    if not (config.QUAD_SOLIDITY_MIN <= solidity <= config.QUAD_SOLIDITY_MAX):
        return False
    aspect = quad_aspect(coords)
    return config.QUAD_ASPECT_MIN <= aspect <= config.QUAD_ASPECT_MAX


def find_largest_contour(image):
    '''
        Picks the sudoku grid contour out of `image`. A bigger object sharing the frame (a
        face, a hand) can easily have a larger contour area than the puzzle itself, so this
        doesn't just take the single largest-area contour - it checks candidates in descending
        area order and returns the first one that actually validates as a well-formed quad,
        letting a smaller-but-rectangular puzzle win over a bigger but irregular shape.

        Returns (quad, coords, is_valid_rect):
          - quad/coords are None if no contour cleared config.CONTOUR_MIN_AREA at all
          - otherwise they're the first candidate (checked largest-area-first) that passed
            validate_rect, or, if none did, the single largest candidate anyway - so callers
            can distinguish "found something, just not rectangular" from "found nothing"
          - is_valid_rect is False whenever no candidate validated
    '''
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) > config.CONTOUR_MIN_AREA]
    if not candidates:
        return None, None, False

    candidates.sort(key=cv2.contourArea, reverse=True)

    fallback_quad, fallback_coords = None, None
    for contour in candidates[:MAX_CONTOUR_CANDIDATES]:
        quad = _simplify_to_quad(contour)
        if quad is None:
            continue
        coords = get_corners(quad)
        if fallback_quad is None:
            fallback_quad, fallback_coords = quad, coords
        if is_grid_quad(contour, coords):
            return quad, coords, True

    return fallback_quad, fallback_coords, False


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
        same search).

        Returns (quad, coords, is_grid) scaled back to frame's original resolution, where
        is_grid is the verdict of the full shape test (see is_grid_quad). Callers should trust
        that flag rather than re-running validate_rect, which on its own is too weak to reject
        an irregular blob. (None, None, False) means nothing cleared the area threshold.
    '''
    small, scale = _downscale(frame, config.DETECTION_MAX_DIM)
    processed = preprocess(small)
    quad, coords, is_grid = find_largest_contour(processed)
    if quad is None:
        return None, None, False

    if scale != 1.0:
        quad = np.round(quad.astype(np.float32) / scale).astype(np.int32)
        coords = coords / scale

    return quad, coords, is_grid


def _line_extent(line_mask, axis):
    '''
        Positions of the first and last strong ruled line along one axis, or None if too few
        lines are visible to trust the result.
    '''
    profile = line_mask.sum(axis=axis).astype(np.float32)
    peak = profile.max()
    if peak <= 0:
        return None
    strong = np.where(profile > 0.35 * peak)[0]
    if len(strong) == 0:
        return None

    groups = []
    for index in strong:
        if groups and index - groups[-1][-1] <= 6:
            groups[-1].append(index)
        else:
            groups.append([index])
    if len(groups) < 4:  # need several ruled lines before trusting the extent
        return None
    return int(np.mean(groups[0])), int(np.mean(groups[-1]))


def refine_grid_bounds(warped_inv):
    '''
        Corrects a warp whose quad overshot the puzzle.

        The outer contour often merges the grid with whatever sits beside it on the page or
        screen, so the warped square can include a margin of unrelated content - which shifts
        every cell boundary and makes the fixed 1/9 slicing sample across cell edges. The
        puzzle's own ruling is the reliable landmark: this finds the outermost ruled lines and
        returns the crop bounding them, or None when too few lines are visible to be sure.
    '''
    ink = (warped_inv < 128).astype(np.uint8) * 255
    height, width = ink.shape[:2]

    h_len = max(3, int(width * config.GRID_LINE_LENGTH_RATIO))
    v_len = max(3, int(height * config.GRID_LINE_LENGTH_RATIO))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))

    rows = _line_extent(horizontal, 1)
    cols = _line_extent(vertical, 0)
    if rows is None or cols is None:
        return None

    y0, y1 = rows
    x0, x1 = cols
    # a refinement that keeps almost nothing, or changes almost nothing, isn't worth applying
    if (y1 - y0) < 0.5 * height or (x1 - x0) < 0.5 * width:
        return None
    if (y1 - y0) > 0.98 * height and (x1 - x0) > 0.98 * width:
        return None
    return x0, y0, x1, y1


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
