import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import cv2 as cv
import glob
from time import sleep
from multiprocessing import Process, Manager


def process_image(L,params, i):
    objpoints, imgpoints, gray_shape = params
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)
    L[i] = [ret, mtx, dist, rvecs, tvecs]




if __name__ == '__main__':

    with Manager() as manager:
        with open('outputs/objpoints_imgpoints_gray-shape.pkl', 'rb') as f:
            params = pickle.load(f)

        objpoints, imgpoints, gray_shape = params

        assert len(objpoints) == len(imgpoints), ' size of object point and image points do not match'


        # Generate parameters using lots of subset for plotting purposes


        params_subsets = []

        number_of_subsets = 1000
        number_of_elem_in_subset = 20

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

    fx_extracted = []

    for params in camera_params_by_subsets:
        mtx= np.array(params[1])
        fx_extracted.append(mtx[0][0])

    plt.hist(fx_extracted, density=True, bins=100)  # `density=False` would make counts
    plt.ylabel('Probability')
    plt.xlabel('Data');
    plt.savefig('./outputs/'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_fx.png')

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
    # plt.show(block=True)

    # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    # print(ret, mtx, dist, rvecs, tvecs)


    # Params
    #  [[3.39130655e+03 0.00000000e+00 1.98395321e+03]
    #  [0.00000000e+00 3.38196140e+03 1.50596553e+03]
    #  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
