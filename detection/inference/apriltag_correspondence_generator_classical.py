"""
This code assumes that images used for calibration are of the same arUco marker board provided with code
"""

import cv2
from cv2 import aruco
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# root directory of repo for relative path specification.
root = Path(__file__).parent.absolute()
# resize_shape = (1024*2 , 1024*2)

# Set path to the images
calib_imgs_path = root.joinpath("img/Calibrate_5x5_1")

# For validating results, show aruco board to camera.
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_APRILTAG_36h11 )

#Provide length of the marker's side
markerLength = 2.4  # Here, measurement unit is centimetre.

# Provide separation between markers
markerSeparation = 1.2   # Here, measurement unit is centimetre.


# create arUco board
board = aruco.GridBoard_create(5, 5, markerLength, markerSeparation, aruco_dict, firstMarker = 300)

'''uncomment following block to draw and show the board'''
#img = board.draw((864,1080))
#cv2.imshow("aruco", img)

arucoParams = aruco.DetectorParameters_create()
img_list = []
calib_fnms = calib_imgs_path.glob('*.jpg')
print('Using ...', end='')
for idx, fn in enumerate(calib_fnms):

    if(idx>=10):
        break
    print(idx, '', end='')
    img = cv2.imread( str(root.joinpath(fn) ))
    img_list.append( img )
    h, w, c = img.shape
print('Calibration images')

counter, corners_list, id_list = [], [], []
first = True
for im in tqdm(img_list):
    img_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img_gray, aruco_dict, parameters=arucoParams)
    # if first == True: corners_list = corners
    #     id_list = ids
    #     first = False
    # else:
    corners_list = corners_list.append( corners)
    id_list = corners_list.append(ids)
    counter.append(len(ids))
print('Found {} unique markers'.format(np.unique(ids)))
counter = np.array(counter)
print ("Calibrating camera .... Please wait...")
#mat = np.zeros((3,3), float)
ret, mtx, dist, rvecs, tvecs,_, _, reprojectionError = aruco.calibrateCameraArucoExtended(corners_list, id_list, counter, board, img_gray.shape, None, None )
print(mtx)
print(reprojectionError)

# # data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
with open("outputs/corners-list_id-list_counter_orig_testing.pkl", "wb") as f:
    pickle.dump((corners_list, id_list, counter,  img_gray.shape), f)
