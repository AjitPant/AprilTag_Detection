import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import cv2 as cv
import glob
from time import sleep
from multiprocessing import Process, Manager



def process_image(L,params, i):
    corners, id, counter,board, gray_shape = params

    corners = [item for sublist in corners for item in sublist]
    id = [item for sublist in id for item in sublist]
    corners = np.array(corners,dtype = np.float32).reshape((4*len(corners), 2))
    id = np.array(id).reshape(len(corners)//4)
    counter = np.array(counter)

    print(corners)
    print(id)
    print(counter)


    ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv.aruco.calibrateCameraArucoExtended(corners, id, counter,board,  gray_shape, None, None, criteria =       (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 300, 0.01))

    print(perViewErrors)
    L[i] = [ret, mtx, dist, rvecs, tvecs]




if __name__ == '__main__':

    with Manager() as manager:
        with open('outputs/corners-list_id-list_counter_orig_testing.pkl', 'rb') as f:
            params = pickle.load(f)

        corners_list, id_list, counter, gray = params


        print(len(corners_list))
        print(len(id_list))
        print(gray)
        print(len(counter))

        assert len(corners_list) == len(id_list), ' size of corners and id do not match'
        assert len(corners_list) == len(counter), ' size of corners and counter do not match'

        # For validating results, show aruco board to camera.
        aruco_dict = cv.aruco.getPredefinedDictionary( cv.aruco.DICT_APRILTAG_36h11 )

        #Provide length of the marker's side
        markerLength = 2.4  # Here, measurement unit is centimetre.

        # Provide separation between markers
        markerSeparation = 1.2   # Here, measurement unit is centimetre.

        board = cv.aruco.GridBoard_create(5, 5, markerLength, markerSeparation, aruco_dict, firstMarker = 300)
        print(board.objPoints)
        assert(False)

        # Generate parameters using lots of subset for plotting purposes


        params_subsets = []

        number_of_subsets = 1000
        number_of_elem_in_subset =10

        zipped_corner_id_counter = list(zip(corners_list, id_list, counter))
        print(len(zipped_corner_id_counter))


        for _ in range(number_of_subsets):
            print(_)
            # random.shuffle(zipped_corner_id_counter)
            corner_sub, id_sub, counter_sub = zip(*(zipped_corner_id_counter))
            params_subsets.append([corner_sub, id_sub, counter_sub,board, gray])



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

    fx_extracted = []

    for params in camera_params_by_subsets:
        mtx= np.array(params[1])
        fx_extracted.append(mtx[0][0])

    plt.hist(fx_extracted, density=True, bins=100)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data');
    plt.savefig('./outputs/classic_'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_fx_april_300.png')

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
    plt.savefig('./outputs/classic_'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_fy_april_300.png')

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
    plt.savefig('./outputs/classic_'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_cx_april_300.png')

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
    plt.savefig('./outputs/classic_'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_cy_april_300.png')
    # plt.show(block=True)

    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    # print(ret, mtx, dist, rvecs, tvecs)


    # Params
    #  [[3.39130655e+03 0.00000000e+00 1.98395321e+03]
    #  [0.00000000e+00 3.38196140e+03 1.50596553e+03]
    #  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
