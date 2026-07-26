import cv2
import numpy as np


def fill_sudoku(solved, unsolved, img):
    '''
        This funciton is used to fill the warped sudoku image with the solution.
            The funcion expects a solved sudoku matrix, an unsolved sudoku matrix, and the warped image
    '''
    # First we calculate the width and height of the warped image.
    gridw = img.shape[1]
    gridh = img.shape[0]

    #Divide the width and height by 9 to get the block locations
    xgap = gridw // 9
    ygap = gridh // 9
    # added a small margin value to fit the text values a littel more better in their respective blocks
    margin = int(0.015 * img.shape[1])

    for i in range(9):
        for j in range(9):
            #only write those numbers which are solved
            if unsolved[i][j] == 0:
                text = str(solved[i][j])
                xloc = xgap*j + margin
                yloc = ygap*(i+1) - margin
                fontsize = gridw / 400
                cv2.putText(img, text, (xloc, yloc), cv2.FONT_HERSHEY_SIMPLEX, fontsize, (0,165,255), 2)

    return img


def unwarp_image(img_src, img_dest, pts_dest):
    '''
        This function is used to warp the solution image onto the actual frame.
    '''
    pts_dest = np.array(pts_dest)

    height, width = img_src.shape[0], img_src.shape[1]
    pts_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts_dest)
    warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
    cv2.fillConvexPoly(img_dest, pts_dest.astype('int32'), 0)

    dst_img = cv2.add(img_dest, warped)

    return dst_img
