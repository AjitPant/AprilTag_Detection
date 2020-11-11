import numpy as np
import pickle
import cv2 as cv
import glob
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('img/Calibration_test1_9x6_checker_22.9mm/*.jpg')

# cv.namedWindow('img', cv.WINDOW_NORMAL)
for ind, fname in enumerate(images):
    if(ind >=1):
        break
    print(ind, fname)
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    print('before find')
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)
    print('after find')
    # If found, add object points, image points (after refining them)
    if ret == True:
        print('in find')
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        print('before draw')
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        print('after draw')
        # cv.imshow('img', img)
        # cv.waitKey(0)
cv.destroyAllWindows()

print(objpoints)
print(imgpoints)

params = (objpoints, imgpoints, gray.shape[::-1])

with open('outputs/objpoints_imgpoints_gray-shape.pkl', 'wb') as f:
   pickle.dump(params, f)

print("abc")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],cameraMatrix=None,  distCoeffs=np.zeros((5,)), flags=cv.CALIB_FIX_K5)

print("hi")
# print(reprojectionError)
print(ret, mtx, dist, rvecs, tvecs)
print("hixxx")
