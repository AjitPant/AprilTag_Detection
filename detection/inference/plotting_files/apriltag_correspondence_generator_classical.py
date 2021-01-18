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

    files_list = glob.glob(os.path.join(args.image_dir, '*.jpg'))
    print("Found {} files in {}".format(len(files_list),args.image_dir))

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    print("Dictionary Initialzed with family {}", cv2.aruco.DICT_APRILTAG_36h11)

    detectorParameters = cv2.aruco.DetectorParameters_create()
    detectorParameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

    extracted_corners ,extracted_ids, extracted_counter = [], [], []

    print("Detector Parameters Initialzed ")

    for file_name in tqdm(files_list):
        print("Working on ", file_name)
        image = cv2.imread(file_name)
        image = cv2.resize(image, (1024, 1024))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary, parameters=detectorParameters)
        if corners is None or ids is None:
            continue

 #       corners, ids = zip(*[ [corners,_] for _,corners in sorted(zip(ids,corners)) ])


        extracted_corners.append(corners)
        extracted_ids.append(ids)
        extracted_counter.append(len(ids))

        if(args.visualize):
            image =	cv2.aruco.drawDetectedMarkers(	image, corners, ids	)
            cv2.namedWindow(file_name, cv2.WINDOW_NORMAL)
            cv2.imshow(file_name, image)
            cv2.waitKey(0)


    if(args.visualize):
        cv2.destroyAllWindows()

    markerLength = 0.8 # Here, measurement unit is centimetre.
    markerSeparation = 0.3   # Here, measurement unit is centimetre.

    board = cv2.aruco.GridBoard_create(5, 5, markerLength, markerSeparation, dictionary, firstMarker = 300)

    # img = board.draw((500 , 500))
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    formatter = AprilTagHelper.AprilTagFormatter()
    objPoints = formatter(extracted_corners, extracted_ids, board)



    objPoints = objPoints




    corner = []

    for corners in extracted_corners:
        corners = np.vstack(corners).reshape(-1,4, 2).astype(np.float32)
        # corners[:,[0, 1,2, 3], :] = corners[:, [  1, 0, 3, 2], :]
        corners = corners.reshape(-1, 2)
        corner.append(corners)
    corners = corner


    # if(args.visualize):
    #     plotter = AprilTagHelper.AprilTagPlotter()
    #     plotter(objPoints, extracted_ids)
    #     plotter(corners ,extracted_ids)

    with open("./outputs/classical_detections.pkl","wb") as f:
        pickle.dump((corners, objPoints, gray.shape), f)

    retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = 	cv2.calibrateCameraExtended(objPoints,corners, gray.shape, cameraMatrix=None , distCoeffs = None	)



    print(cameraMatrix)
    print(distCoeffs)
    print(perViewErrors)

    if(args.visualize):

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        for ind, file_name in enumerate(tqdm(files_list)):
            image = cv2.imread(file_name)

            imagePoints, jacobian	=	cv2.projectPoints(	objPoints[ind], rvecs[ind], tvecs[ind], cameraMatrix, distCoeffs)
            print(imagePoints)
            for point in imagePoints:
                print(point)
                cv2.circle(image,tuple(point[0]), 3, (0,0,255), -1)


            cv2.imshow("image", image)
            cv2.waitKey(0)






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--image_dir", required=True,type = str,
                        help="path to image directory containing images of AprilTag")
    parser.add_argument("-T", "--tag_family",type = str,
                        help="family of Apriltag (default:tag36h11) ", default = "tag36h11")

    parser.add_argument( "--visualize",type = bool,
                        help="Draw images for debugging", default = False)
    args = parser.parse_args()
    process(args)



if __name__ == "__main__":
    main()
