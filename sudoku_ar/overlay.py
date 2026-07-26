import cv2
import numpy as np

DIGIT_COLOR = (0, 165, 255)


def render_solution_canvas(solved, unsolved, size):
    '''
        Renders the solution digits once onto a canonical size x size canvas, with a matching
        single-channel mask marking which pixels were drawn on. Caching this (instead of
        redrawing text onto a re-warped copy of the live frame every single frame, then
        unwarping the whole thing back) means the per-frame cost while locked is just warping
        + compositing this small cached canvas, and the live grid stays visible at full
        sharpness everywhere except the digits themselves.
    '''
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    xgap = size // 9
    ygap = size // 9
    margin = int(0.015 * size)
    fontsize = size / 400

    for i in range(9):
        for j in range(9):
            if unsolved[i][j] == 0:
                text = str(solved[i][j])
                xloc = xgap*j + margin
                yloc = ygap*(i+1) - margin
                cv2.putText(canvas, text, (xloc, yloc), cv2.FONT_HERSHEY_SIMPLEX, fontsize, DIGIT_COLOR, 2)
                cv2.putText(mask, text, (xloc, yloc), cv2.FONT_HERSHEY_SIMPLEX, fontsize, 255, 2)

    return canvas, mask


def composite(frame, canvas, mask, coords, warp_size):
    '''
        Warps the cached solution canvas/mask from canonical warp-space into the live frame
        using the current detected corners, then alpha-composites it over the frame in place
        - only the (small, cached) overlay is warped, never the live camera image itself.
    '''
    pts_source = np.array([
        [0, 0],
        [warp_size - 1, 0],
        [warp_size - 1, warp_size - 1],
        [0, warp_size - 1]], dtype='float32')

    h, _ = cv2.findHomography(pts_source, np.array(coords, dtype='float32'))
    warped_canvas = cv2.warpPerspective(canvas, h, (frame.shape[1], frame.shape[0]))
    warped_mask = cv2.warpPerspective(mask, h, (frame.shape[1], frame.shape[0]))

    drawn = warped_mask > 0
    frame[drawn] = warped_canvas[drawn]
    return frame
