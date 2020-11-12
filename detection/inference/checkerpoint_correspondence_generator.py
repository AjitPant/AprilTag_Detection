import numpy as np
import pickle
import cv2
import glob
import os
import argparse


def process(args):
    ncols = args.nsquare_x - 1
    nrows = args.nsquare_y - 1

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objpts = np.zeros((ncols * nrows, 3), np.float32)
    objpts[:, :2] = np.mgrid[0:ncols, 0:nrows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpts_list = list()  # 3d point in real world space
    imgpts_list = list()  # 2d points in image plane.
    images = glob.glob(os.path.join(args.images_dir, "*.jpg"))

    for i, filename in enumerate(images):
        print("{0}) Reading {1}".format(i + 1, filename))
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (ncols, nrows), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpts_list.append(objpts)
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpts_list.append(refined_corners)
            if args.visualize:
                # Draw and display the corners
                scale = 0.25
                draw = cv2.resize(image, None, fx=scale, fy=scale)
                draw = cv2.drawChessboardCorners(draw, (ncols, nrows), refined_corners * scale, ret)
                cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
                cv2.imshow(filename, draw)
                cv2.waitKey(0)

    if args.visualize:
        cv2.destroyAllWindows()

    if args.output_dir is not None:
        with open(os.path.join(args.output_dir, "data.pkl"), "wb") as f:
            data = (objpts_list, imgpts_list, gray.shape[::-1])
            pickle.dump(data, f)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpts_list, imgpts_list, gray.shape[::-1], None, None, flags=cv2.CALIB_FIX_K3)
    print("camera matrix: {}".format(camera_matrix))
    print("distortion coefficients: {}".format(dist_coeffs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images_dir',
        type=str,
        required=True,
        help="Directory contains images.")
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Directory to save results.")
    parser.add_argument(
        '--nsquare_x',
        type=int,
        default=10,
        help="Number of squares in x direction.")
    parser.add_argument(
        '--nsquare_y',
        type=int,
        default=7,
        help="Number of squares in y direction.")
    parser.add_argument(
        '--visualize',
        type=bool,
        default=False,
        help="Visualize detected checkerboard corners.")
    args = parser.parse_args()
    process(args)


if __name__ == '__main__':
    main()
