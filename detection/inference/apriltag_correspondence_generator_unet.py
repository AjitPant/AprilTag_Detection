"""
This code assumes that images used for calibration are of the same arUco marker board provided with code
"""
import cv2
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import glob
import os
import AprilTagHelper


def process(args):
    with open(args.input_pkl, "rb") as f:
        params = pickle.load(f);

    extracted_corners, extracted_ids, counter, gray_shape = params



    markerLength = 2.4 # Here, measurement unit is centimetre.
    markerSeparation = 1.2   # Here, measurement unit is centimetre.

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    board = cv2.aruco.GridBoard_create(5, 5, markerLength, markerSeparation, dictionary, firstMarker = 350)

    # img = board.draw((500 , 500))
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    formatter = AprilTagHelper.AprilTagFormatter()
    objPoints = formatter(extracted_corners, extracted_ids, board)



    objPoints = objPoints




    corner = []

    for corners in extracted_corners:
        if(len(corners) ==0):
            continue
        corners = np.vstack(corners).reshape(-1,4, 2).astype(np.float32)
        # corners[:,[0, 1,2, 3], :] = corners[:, [  1, 0, 3, 2], :]
        corners = corners.reshape(-1, 2)
        corner.append(corners)
    corners = corner


    if(args.visualize):
        plotter = AprilTagHelper.AprilTagPlotter()
        plotter(objPoints, extracted_ids)
        plotter(corners ,extracted_ids)

    with open("./outputs/unet_detections.pkl","wb") as f:
        pickle.dump((corners, objPoints, gray_shape), f)

    retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = 	cv2.calibrateCameraExtended(objPoints,corners, gray_shape[:2], cameraMatrix=None , distCoeffs = None	)



    print(cameraMatrix)
    print(distCoeffs)
    print(perViewErrors)

    # if(args.visualize):

    #     cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    #     for ind, file_name in enumerate(tqdm(files_list)):
    #         image = cv2.imread(file_name)

    #         imagePoints, jacobian	=	cv2.projectPoints(	objPoints[ind], rvecs[ind], tvecs[ind], cameraMatrix, distCoeffs)
    #         print(imagePoints)
    #         for point in imagePoints:
    #             print(point)
    #             cv2.circle(image,tuple(point[0]), 3, (0,0,255), -1)


    #         cv2.imshow("image", image)
    #         cv2.waitKey(0)






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input_pkl", required=True,type = str,
                        help="Path to input pkl file")
    parser.add_argument( "--visualize",type = bool,
                        help="Draw images for debugging", default = False)
    args = parser.parse_args()
    process(args)



if __name__ == "__main__":
    main()
