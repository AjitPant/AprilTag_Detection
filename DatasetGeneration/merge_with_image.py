
from scipy.spatial import distance as dist
import time
import random
import os
import copy
import argparse
import cv2
import numpy as np
from apriltag_images import TAG36h11, AprilTagImages
from apriltag_generator import AprilTagGenerator
from backgound_overlayer import backgroundOverlayer

import multiprocessing
from multiprocessing import Pool
from multiprocessing import freeze_support
from threading import Lock
import itertools

from scipy import stats
import operator
import math
from functools import reduce

import torch


def nonzero_mode(arr):
    return stats.mode(arr[np.nonzero(arr)]).mode

from scipy.spatial import distance as dist
import numpy as np
import cv2
def order_points(pts):
	coords = pts
	center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
	out = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
	return np.array(out, dtype="float32")
mutex = Lock()


def reduce_to_tags(img, response_1, response_2, filename, args):
    mask_segmentation = response_1
    mask_corners = response_2
    segregates = []

    mask_corners =  np.argmax(mask_corners, axis=2)

    # cv2.namedWindow('mask_segmentation', cv2.WINDOW_NORMAL)
    # cv2.imshow("mask_segmentation", mask_segmentation)

    # cv2.namedWindow('mask_garbage_0', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('mask_garbage_1', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('mask_garbage_2', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('mask_garbage_3', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('mask_garbage_4', cv2.WINDOW_NORMAL)

    mask_real_corners = np.zeros(mask_corners.shape[1:], dtype=np.uint8)
    # print(mask_corners.max())
    # for i in range(4):
    #     mask_real_corners +=mask_corners[i]*(i+1)
    mask_real_corners = (mask_corners!=4).astype(np.uint8)
    # mask_real_corners = 4- np.argmax(mask_corners, axis = 0).astype(np.uint8)
    # print(mask_real_corners)

    # cv2.namedWindow('mask_garbage', cv2.WINDOW_NORMAL)
    # cv2.imshow("mask_garbage", mask_real_corners.astype(np.float32)*60)
    # cv2.waitKey(0)

    contours, _ = cv2.findContours(
        mask_segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    temp_img = img.copy()
    coords = np.argwhere(mask_corners > 0)

    cv2.drawContours(temp_img, contours, -1, (0, 255, 0), 3)

    # cv2.namedWindow('contours_img', cv2.WINDOW_NORMAL)
    # cv2.imshow("contours_img", temp_img)
    # cv2.waitKey(0)
    index = 0

    for ind in range(len(contours)):
        segregates = []
        internal_mask = np.zeros(mask_real_corners.shape, dtype=np.uint8)

        cv2.drawContours(internal_mask, contours, ind, 255, -1)

        # cv2.namedWindow('internal_mask', cv2.WINDOW_NORMAL)
        # cv2.imshow("internal_mask", internal_mask)
        # cv2.waitKey(0)
        # print("hello")
        internal_mask = cv2.bitwise_and(
            internal_mask, mask_real_corners.astype(np.uint8))
        # print("heello")

        internal_contours, _ = cv2.findContours(
            internal_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print("heeello")
        # print(len(internal_contours))
        for inner_ind in range(len(internal_contours)):
            internal_internal_mask = np.zeros(
                mask_real_corners.shape, dtype=np.uint8)
            cv2.drawContours(internal_internal_mask,
                             internal_contours, inner_ind, 255, -1)
            internal_internal_mask = cv2.bitwise_and(
                internal_internal_mask, mask_real_corners.astype(np.uint8))

            mode = map(nonzero_mode, internal_internal_mask)
            #find the center of contours
            M = cv2.moments(internal_contours[inner_ind])
            cX = int(M["m10"] / (M["m00"]+1e-9))
            cY = int(M["m01"] / (M["m00"]+1e-9))
            segregates.append([cX, cY])
        print(segregates)
        if len(segregates) != 4:
            continue
        segregates = order_points(segregates)
        # print(len(segregates))
        if len(segregates) != 4:
            continue

        corner_list = []
        # print(segregates)
        for i in segregates:
            corner_list.append((i[0], i[1]))

        assert len(corner_list) == 4
        # print(corner_list)
        rand = random.randrange(1,10)
        h, status = cv2.findHomography(
            np.array(corner_list), np.array([[0+rand, 0+rand], [0+rand, 224-rand], [224-rand, 224-rand], [224-rand, 0+rand]]))
        height, width, channels = img.shape
        im1Reg = cv2.warpPerspective(img, h, (224, 224))
        # cv2.namedWindow('a', cv2.WINDOW_NORMAL)
        # cv2.imshow('a', im1Reg)
        # cv2.waitKey(0)
        # print(corner_list)

        cv2.imwrite(os.path.join(args.out_folder, 'simg',
                                 filename[:-4] + "_" + str(index) + '.jpg'), im1Reg)


        print(f"{mask_corners[int(corner_list[0][1])][int(corner_list[0][0])]}")
        if mask_corners[int(corner_list[0][1])][int(corner_list[0][0])] == 4:
            assert(False)
        with open(os.path.join(args.out_folder, 'simg',filename[:-4] + "_" + str(index) + '.txt'), "w") as text_file:
            print(f"{mask_corners[int(corner_list[0][1])][int(corner_list[0][0])]}", file=text_file)

        index = index + 1







def augment_and_save(file, overlayer, args):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".bmp"):
        path = (os.path.join(args.img_folder, filename))
        for j in range(1):
            img = cv2.imread(path)

            if img is None:
                print("Failed to load the {}. Make sure it exists.", path)
                exit()

            img = cv2.resize(img, (512*2, 512*2))
            img_out, response_1, response_2 = overlayer(img)
            # reduce_to_tags(img_out, response_1, response_2, filename, args)

            img_out = cv2.resize(img_out, (512, 512), interpolation=cv2.INTER_AREA)
            response_1 = cv2.resize(response_1, (512, 512), interpolation=cv2.INTER_AREA)
            response_2 = cv2.resize(response_2, (512, 512), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(args.out_folder, 'img',
                                     filename[:-4] + "_" + str(j) + '.jpg'), img_out)
            cv2.imwrite(os.path.join(args.out_folder, 'mask',
                                     filename[:-4] + "_" + str(j) + '_5.png'), response_1)
            for k in range(1):
                cv2.imwrite(os.path.join(args.out_folder, 'mask',  filename[:-4] + "_" + str(
                    j) + '_'+str(k) + '.png'), response_2[:, :, k])


def run_multiprocessing(func, file_list, overlayer, args, n_processors):
    parameters = ((file, overlayer, args) for file in file_list)
    with Pool(processes=n_processors) as pool:
        return pool.starmap(func, parameters)


def app():
    parser = argparse.ArgumentParser(description='April tag image Generator.')
    parser.add_argument(
        '--root',
        type=str,
        default='.',
        help='Directory to all standard April tag images.')
    parser.add_argument(
        '--img_folder',
        type=str,
        default='./imgs',
        help='Folder which contains background images')
    parser.add_argument(
        '--out_folder',
        type=str,
        default='./out',
        help='Output folder which contains dataset')
    parser.add_argument(
        '--family',
        type=str,
        default=TAG36h11,
        help='April tag family.')
    parser.add_argument(
        '--size',
        type=int,
        default=1024,
        help='Size of April tag images in pixels.')
    parser.add_argument(
       '--mx_tags',
        type=int,
        default=30,
        help='Maximum number of tags to generate in an image')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_folder, 'img'), exist_ok=True)
    os.makedirs(os.path.join(args.out_folder, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(args.out_folder, 'simg'), exist_ok=True)

    # logger = multiprocessing.log_to_stderr()
    # logger.setLevel(multiprocessing.SUBDEBUG)
    generator = AprilTagGenerator(root=args.root,
                                  family=args.family,
                                  size=args.size,
                                  rx_lim_deg=(30, 70),
                                  ry_lim_deg=(30, 70),
                                  rz_lim_deg=(-180, 180),
                                  scalex_lim=(0.125/2/1.5, 1.0/8),
                                  scaley_lim=(0.125/2/1.5, 1.0/8),
                                  )

    print(len(generator))
    overlayer = backgroundOverlayer(generator, args.mx_tags)
    directory = os.fsencode(args.img_folder)
    i = 0

    n_processors = 4

    mx_files = 500

    file_list = sorted(list(os.listdir(directory))[mx_files//2:mx_files])

    '''
    pass the task function, followed by the parameters to processors
    '''
    start = time.time()
    out = run_multiprocessing(
        augment_and_save, file_list, overlayer, args, n_processors)
    print("Mutiprocessing time: {}secs\n".format((time.time()-start)))


if __name__ == "__main__":
    freeze_support()   # required to use multiprocessing
    app()
