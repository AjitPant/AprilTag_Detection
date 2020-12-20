import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import cv2 as cv
import glob
from time import sleep
from multiprocessing import Process, Manager


def process_image(L,params, i):
    # img= cv.imread('img/calibration_temp/IMG_20201118_023654.jpg')
    objpoints, imgpoints, gray_shape = params
    # ret, mtx, dist, rvecs, tvecs,_,_, reprojectionError = cv.calibrateCameraExtended(objpoints, imgpoints, gray_shape, np.array([[3600, 0, 1400], [0, 3600, 1400], [0,0,1.0]]), distCoeffs =np.array([ 2.76365982e-01, -1.09454562e+00,  3.26174409e-04,  1.51208483e-03 ,1.39965460e+00]),flags=cv.CALIB_USE_INTRINSIC_GUESS  )
    ret, mtx, dist, rvecs, tvecs,_,_, reprojectionError = cv.calibrateCameraExtended(objpoints, imgpoints, gray_shape[::-1], None, None  )
    L[i] = [ret, mtx, dist, rvecs, tvecs, reprojectionError]
    # # undistort
    # dst = cv.undistort(img, mtx, dist)
    # # crop the image
    # cv.namedWindow('calibresult.png', cv.WINDOW_NORMAL)
    # cv.imshow('calibresult.png', dst)
    # cv.waitKey(0)

    # print(dist)




if __name__ == '__main__':

    with Manager() as manager:
        with open('outputs/chessboard_bmp.pkl', 'rb') as f:
            params = pickle.load(f)

        objpoints, imgpoints, gray_shape = params
        print(objpoints)

        # gray_shape = gray_shape[::-1]
        assert len(objpoints) == len(imgpoints), ' size of object point and image points do not match'


        print(len(objpoints))


        # Generate parameters using lots of subset for plotting purposes


        params_subsets = []

        number_of_subsets =1000
        number_of_elem_in_subset =10


        zipped_objpoints_imgpoints = list(zip(objpoints, imgpoints))



        for _ in range(number_of_subsets):
            print(_)
            objpoints_sub, imgpoints_sub = zip(*random.sample(zipped_objpoints_imgpoints, number_of_elem_in_subset))
            params_subsets.append([objpoints_sub, imgpoints_sub, gray_shape])



        manager_camera_params_by_subsets = manager.list(range(number_of_subsets))  # <-- can be shared between processes.
        processes = []
        for i in range(number_of_subsets):
            p = Process(target=process_image, args=(manager_camera_params_by_subsets,params_subsets[i], i))  # Passing the list
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        camera_params_by_subsets = []
        for ind in range(number_of_subsets):
            camera_params_by_subsets.append(manager_camera_params_by_subsets[ind])

    with open("./outputs/checkerboard_params.pkl", "wb") as f:
        pickle.dump(camera_params_by_subsets, f)
    fx_extracted = []

    for params in camera_params_by_subsets:
        mtx= np.array(params[1])
        fx_extracted.append(mtx[0][0])

    plt.hist(fx_extracted, density=True, bins=100)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data');
    plt.savefig('./outputs/'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_fx.png')
    # plt.show(block = True)

    plt.clf()
    plt.cla()
    plt.close()



    fy_extracted = []

    for params in camera_params_by_subsets:
        mtx= np.array(params[1])
        fy_extracted.append(mtx[1][1])

    plt.hist(fy_extracted, density=True, bins=100)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data');
    plt.savefig('./outputs/'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_fy.png')
    # plt.show(block = True)

    plt.clf()
    plt.cla()
    plt.close()

    cx_extracted = []

    for params in camera_params_by_subsets:
        mtx= np.array(params[1])
        cx_extracted.append(mtx[0][2])

    plt.hist(cx_extracted, density=True, bins=100)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data');
    plt.savefig('./outputs/'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_cx.png')
    # plt.show(block = True)

    plt.clf()
    plt.cla()
    plt.close()

    cy_extracted = []

    for params in camera_params_by_subsets:
        mtx= np.array(params[1])
        cy_extracted.append(mtx[1][2])

    plt.hist(cy_extracted, density=True, bins=100)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data');
    plt.savefig('./outputs/'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_cy.png')
    # plt.show(block = True)

    plt.clf()
    plt.cla()
    plt.close()

    reprojectionError_extracted = []

    for params in camera_params_by_subsets:
        error= np.array(params[5])
        reprojectionError_extracted.append(np.mean(error))


    plt.hist(reprojectionError_extracted, density=True, bins=100)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data');
    plt.savefig('./outputs/'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_reprojectionError.png')


    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    # print(ret, mtx, dist, rvecs, tvecs)


    # Params
    #  [[3.39130655e+03 0.00000000e+00 1.98395321e+03]
    #  [0.00000000e+00 3.38196140e+03 1.50596553e+03]
    #  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
