
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import cv2 as cv
import glob
from time import sleep
from multiprocessing import Process, Manager



def process_image(L,params, i):
    corners, objPoints, gray_shape = params

    ret, mtx, dist, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv.calibrateCameraExtended(objPoints, corners,  gray_shape[:2], None, None)

    L[i] = [ret, mtx, dist, rvecs, tvecs, perViewErrors]
    print(perViewErrors)





if __name__ == '__main__':

    lb = 2.40
    ub = 2.401
    for iota in np.arange(lb, ub, 0.01):

        with Manager() as manager:
            with open('outputs/unet_detections.pkl', 'rb') as f:
                params = pickle.load(f)

            corners_list, objPoints_list,  gray_shape = params


            assert len(corners_list) == len(objPoints_list), ' size of corners and id do not match'



            params_subsets = []

            number_of_subsets = 1000
            number_of_elem_in_subset =10
            zipped_corner_obj_points = list(zip(corners_list, objPoints_list))


            for _ in range(number_of_subsets):
                print(_)
                corner_sub, objPoints_sub = zip(*random.sample(zipped_corner_obj_points, number_of_elem_in_subset))
                params_subsets.append([corner_sub, objPoints_sub, gray_shape])



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

        with open("./outputs/unet_params.pkl", "wb") as f:
            pickle.dump(camera_params_by_subsets, f)

        fx_extracted = []

        for params in camera_params_by_subsets:
            mtx= np.array(params[1])
            fx_extracted.append(mtx[0][0])

        plt.hist(fx_extracted, density=True, bins=100)  # `density=False` would make counts
        plt.ylabel('Probability')
        plt.xlabel('Data');
        plt.savefig('./outputs/unet_'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_fx_april_350_' + str(2.4) + '.png')

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
        plt.savefig('./outputs/unet_'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_fy_april_350_' + str(2.4) + '.png')

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
        plt.savefig('./outputs/unet_'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_cx_april_350_'+ str(2.4) + '.png')

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
        plt.savefig('./outputs/unet_'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_cy_april_350_'+ str(2.4) + '.png')
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
        plt.savefig('./outputs/unet_'+str(number_of_subsets)+'_'  + str(number_of_elem_in_subset) +'_reprojectionError_april_350_'+ str(2.4) + '.png')
        plt.clf()
        plt.cla()
        plt.close()



        # plt.show(block=True)

        # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

        # print(ret, mtx, dist, rvecs, tvecs)


        # Params
        #  [[3.39130655e+03 0.00000000e+00 1.98395321e+03]
        #  [0.00000000e+00 3.38196140e+03 1.50596553e+03]
        #  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
