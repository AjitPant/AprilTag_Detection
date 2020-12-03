import cv2
from cv2 import aruco
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

# root directory of repo for relative path specification.
root = Path(__file__).parent.absolute()

# Set this flsg True for calibrating camera and False for validating results real time
calibrate_camera = True

# Set path to the images
calib_imgs_path = root.joinpath("./")

# For validating results, show aruco board to camera.
aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_APRILTAG_36h11 )

#Provide length of the marker's side
markerLength = 2.4  # Here, measurement unit is centimetre.

# Provide separation between markers
markerSeparation = 1.2   # Here, measurement unit is centimetre.

# create arUco board
board = aruco.GridBoard_create(5, 5, markerLength, markerSeparation, aruco_dict, 300)

'''uncomment following block to draw and show the board'''
#img = board.draw((864,1080))
#cv2.imshow("aruco", img)

arucoParams = aruco.DetectorParameters_create()

if calibrate_camera == True:
    img_list = []
    calib_fnms = calib_imgs_path.glob('./img/C/*.jpg')
    print('Using ...', end='')
    for idx, fn in enumerate(calib_fnms):
        if(idx>=10):
            break;
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
        if first == True:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list,ids))
        counter.append(len(ids))

    print('Found {} unique markers'.format(np.unique(id_list)))

    counter = np.array(counter)
    print ("Calibrating camera .... Please wait...")
    #mat = np.zeros((3,3), float)

    print(corners_list)
    print(id_list)
    print(counter)

    ret, mtx, dist, rvecs, tvecs = aruco.calibrateCameraAruco(corners_list, id_list, counter, board, img_gray.shape, None, None )

    print("Camera matrix is \n", mtx, "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
    data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open("calibration.yaml", "w") as f:
        yaml.dump(data, f)
