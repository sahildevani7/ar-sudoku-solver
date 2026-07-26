import numpy as np
import cv2

from sudoku_ar import config, detector


def make_frame_with_quad(corners, size=(1280, 720), background=230, line_color=(20, 20, 20)):
    '''
    A camera frame's grid boundary is a thin dark line against a lighter background - draw an
    outline (not a filled shape) so detector.preprocess's small-window adaptive threshold, tuned
    for that contrast pattern, actually picks it up the way it would a real grid.
    '''
    w, h = size
    frame = np.full((h, w, 3), background, dtype=np.uint8)
    pts = np.array(corners, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=line_color, thickness=6)
    return frame


def closest_match_distance(coords, expected):
    '''Each recovered corner should be near some expected corner, regardless of ordering.'''
    distances = []
    for point in coords:
        d = min(np.linalg.norm(point - e) for e in expected)
        distances.append(d)
    return max(distances)


def test_recovers_corners_of_an_axis_aligned_quad():
    corners = [[100, 100], [700, 100], [700, 500], [100, 500]]
    frame = make_frame_with_quad(corners)

    biggest, coords = detector.find_grid(frame)
    assert biggest is not None
    assert detector.validate_rect(coords)
    assert closest_match_distance(coords, np.array(corners, dtype=np.float32)) < 10


def test_recovers_corners_of_a_skewed_quad():
    # a perspective-skewed quad, as a camera at an angle would see the grid
    corners = [[300, 150], [950, 180], [980, 650], [280, 600]]
    frame = make_frame_with_quad(corners)

    biggest, coords = detector.find_grid(frame)
    assert biggest is not None
    assert detector.validate_rect(coords)
    assert closest_match_distance(coords, np.array(corners, dtype=np.float32)) < 10


def test_rejects_a_non_quadrilateral_shape():
    frame = np.full((720, 1280, 3), 230, dtype=np.uint8)
    cv2.circle(frame, (640, 360), 250, (20, 20, 20), 6)

    biggest, coords = detector.find_grid(frame)
    assert biggest is None
    assert coords is None


def test_no_shape_in_frame_finds_nothing():
    frame = np.full((720, 1280, 3), 230, dtype=np.uint8)
    biggest, coords = detector.find_grid(frame)
    assert biggest is None
    assert coords is None


def test_perspective_transform_produces_fixed_warp_size():
    corners = np.array([[300, 150], [950, 180], [980, 650], [280, 600]], dtype=np.float32)
    frame = make_frame_with_quad(corners)

    warped = detector.perspective_transform(corners, frame)
    assert warped.shape[:2] == (config.WARP_SIZE, config.WARP_SIZE)


def test_downscaled_detection_matches_full_resolution_detection():
    '''find_grid downscales large frames for speed - corners recovered should be consistent
    regardless of the frame's native resolution.'''
    corners_small = [[150, 75], [475, 90], [490, 325], [140, 300]]
    corners_large = [[c * 2 for c in point] for point in corners_small]

    frame_small = make_frame_with_quad(corners_small, size=(640, 360))
    frame_large = make_frame_with_quad(corners_large, size=(1280, 720))

    _, coords_small = detector.find_grid(frame_small)
    _, coords_large = detector.find_grid(frame_large)

    assert coords_small is not None and coords_large is not None
    assert closest_match_distance(coords_large / 2.0, coords_small) < 5
