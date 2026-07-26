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

    biggest, coords, is_grid = detector.find_grid(frame)
    assert biggest is not None
    assert is_grid
    assert closest_match_distance(coords, np.array(corners, dtype=np.float32)) < 10


def test_recovers_corners_of_a_skewed_quad():
    # a perspective-skewed quad, as a camera at an angle would see the grid
    corners = [[300, 150], [950, 180], [980, 650], [280, 600]]
    frame = make_frame_with_quad(corners)

    biggest, coords, is_grid = detector.find_grid(frame)
    assert biggest is not None
    assert is_grid
    assert closest_match_distance(coords, np.array(corners, dtype=np.float32)) < 10


def test_rejects_an_asymmetric_shape():
    # An irregular blob - a hand, a shadow, a face silhouette. This must never be accepted as a
    # grid: its contour doesn't simplify to 4 points, so it's discarded before the shape gates.
    frame = np.full((720, 1280, 3), 230, dtype=np.uint8)
    pts = np.array([[400, 200], [900, 220], [850, 600], [600, 650], [420, 500], [380, 350]], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (20, 20, 20))

    biggest, coords, is_grid = detector.find_grid(frame)
    assert not is_grid


def test_rejects_a_rounded_blob_that_passes_the_weak_side_length_check():
    '''
    Regression test for the face-detection bug: a rounded silhouette's four extremal points
    form a quad that validate_rect (which only compares opposite side lengths) happily accepts.
    The solidity/aspect gates are what actually reject it.
    '''
    frame = np.full((720, 1280, 3), 235, dtype=np.uint8)
    angles = np.linspace(0, 2 * np.pi, 60)
    rng = np.random.RandomState(3)
    radii = 240 + rng.normal(0, 28, 60)
    pts = np.stack([(640 + radii * np.cos(angles) * 0.75).astype(int),
                    (360 + radii * np.sin(angles)).astype(int)], axis=1)
    cv2.fillPoly(frame, [pts], (25, 25, 25))

    _, _, is_grid = detector.find_grid(frame)
    assert not is_grid


def test_picks_a_smaller_valid_quad_over_a_bigger_invalid_shape():
    # Regression test: a bigger object sharing the frame (a face, a hand) must not beat a
    # smaller-but-actually-rectangular puzzle just because it has more contour area.
    frame = np.full((720, 1280, 3), 230, dtype=np.uint8)

    big_asymmetric_blob = np.array(
        [[400, 200], [900, 220], [850, 600], [600, 650], [420, 500], [380, 350]], dtype=np.int32)
    cv2.fillPoly(frame, [big_asymmetric_blob], (20, 20, 20))
    assert cv2.contourArea(big_asymmetric_blob) > 150000  # much bigger than the puzzle below

    puzzle_corners = [[950, 50], [1200, 60], [1210, 350], [940, 340]]
    pts = np.array(puzzle_corners, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(20, 20, 20), thickness=6)

    quad, coords, is_grid = detector.find_grid(frame)
    assert quad is not None
    assert is_grid
    assert closest_match_distance(coords, np.array(puzzle_corners, dtype=np.float32)) < 15


def test_finds_grid_next_to_a_larger_rounded_silhouette():
    '''
    The reported failure mode: a face in shot produced a larger contour than the puzzle, so
    detection tracked the face and fed hair/skin texture to the digit recogniser. A real grid
    must win even when something bigger shares the frame.
    '''
    frame = np.full((720, 1280, 3), 235, dtype=np.uint8)

    angles = np.linspace(0, 2 * np.pi, 60)
    rng = np.random.RandomState(3)
    radii = 300 + rng.normal(0, 30, 60)
    silhouette = np.stack([(950 + radii * np.cos(angles) * 0.75).astype(int),
                           (360 + radii * np.sin(angles)).astype(int)], axis=1)
    cv2.fillPoly(frame, [silhouette], (25, 25, 25))

    gx, gy, gs = 80, 180, 330
    cv2.rectangle(frame, (gx, gy), (gx + gs, gy + gs), (20, 20, 20), 4)
    for i in range(1, 9):  # internal cell lines, as a real puzzle has
        offset = int(gs * i / 9)
        cv2.line(frame, (gx + offset, gy), (gx + offset, gy + gs), (20, 20, 20), 2)
        cv2.line(frame, (gx, gy + offset), (gx + gs, gy + offset), (20, 20, 20), 2)

    assert cv2.contourArea(silhouette) > gs * gs  # the distractor really is bigger

    _, coords, is_grid = detector.find_grid(frame)
    assert is_grid
    expected = np.array([[gx, gy], [gx + gs, gy], [gx + gs, gy + gs], [gx, gy + gs]], dtype=np.float32)
    assert closest_match_distance(coords, expected) < 10


def test_no_shape_in_frame_finds_nothing():
    frame = np.full((720, 1280, 3), 230, dtype=np.uint8)
    biggest, coords, is_grid = detector.find_grid(frame)
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

    _, coords_small, _ = detector.find_grid(frame_small)
    _, coords_large, _ = detector.find_grid(frame_large)

    assert coords_small is not None and coords_large is not None
    assert closest_match_distance(coords_large / 2.0, coords_small) < 5
