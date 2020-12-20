import numpy as np
import pickle
import cv2
import glob
import os
import argparse
from multiprocessing import Process, Manager



def parallization_function(args, i, filename, global_objpts, global_imgpts, global_mark):
    ncols = args.nsquare_x - 1
    nrows = args.nsquare_y - 1

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objpts = np.zeros((ncols * nrows, 3), np.float32)
    objpts[:, :2] = np.mgrid[0:ncols, 0:nrows].T.reshape(-1, 2)*2.29

    print("{0}) Reading {1}".format(i + 1, filename))
    image = cv2.imread(filename)
    image = cv2.resize(image, (2048, 2048))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    # if(height>width):
    #     gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)


    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (ncols, nrows), None)

    # If found, add object points, image points (after refining them)
    if ret:
        global_objpts[i] = (objpts)
        refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        global_imgpts[i] = refined_corners
        global_mark[i] = True
        if args.visualize:
            # Draw and display the corners
            scale = 1.0
            draw = cv2.resize(image, None, fx=scale, fy=scale)
            draw = cv2.drawChessboardCorners(draw, (ncols, nrows), refined_corners * scale, ret)
            cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
            cv2.imshow(filename, draw)
            cv2.waitKey(0)
    print("{0}) Completed {1}".format(i + 1, filename))




def process(args):
    # Arrays to store object points and image points from all the images.
    objpts_list = list()  # 3d point in real world space
    imgpts_list = list()  # 2d points in image plane.
    images = glob.glob(os.path.join(args.images_dir, "*.bmp"))


    with Manager() as manager:


        global_objpts = manager.list(range(len(images)))  # <-- can be shared between processes.
        global_imgpts = manager.list(range(len(images)))  # <-- can be shared between processes.
        global_mark = manager.list([False for _ in range(len(images))])  # <-- can be shared between processes.

        processes = []
        for i, filename in enumerate(images):
            p = Process(target=parallization_function, args=(args, i, filename,global_objpts, global_imgpts, global_mark))  # Passing the list
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        for ind in range(len(images)):
            if global_mark[ind]:
                objpts_list.append(global_objpts[ind])
                imgpts_list.append(global_imgpts[ind])


    if args.visualize:
        cv2.destroyAllWindows()

    for i, filename in enumerate(images):
        print("{0}) Reading {1}".format(i + 1, filename))
        image = cv2.imread(filename)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "chessboard_bmp.pkl"), "wb") as f:
            data = (objpts_list, imgpts_list, gray.shape)
            pickle.dump(data, f)


    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpts_list, imgpts_list, gray.shape, None, None)
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
        default=7,
        help="Number of squares in x direction.")
    parser.add_argument(
        '--nsquare_y',
        type=int,
        default=10,
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
