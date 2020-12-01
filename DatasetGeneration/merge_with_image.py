import pickle
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


cnt = [0,0,0,0]
def reduce_to_tags(img, response_1, response_2,response_id, filename, args):
    mask_segmentation = response_1
    mask_corners = response_2
    segregates = []
    mask_corners =  np.argmax(mask_corners, axis=2)

    mask_real_corners = np.zeros(mask_corners.shape[1:], dtype=np.uint8)

    mask_real_corners = (mask_corners!=4).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    temp_img = img.copy()
    coords = np.argwhere(mask_corners > 0)

    cv2.drawContours(temp_img, contours, -1, (0, 255, 0), 3)

    index = 0

    for ind in range(len(contours)):
        segregates = []
        internal_mask = np.zeros(mask_real_corners.shape, dtype=np.uint8)

        cv2.drawContours(internal_mask, contours, ind, 255, -1)

        internal_mask = cv2.bitwise_and(
            internal_mask, mask_real_corners.astype(np.uint8))

        internal_contours, _ = cv2.findContours(
            internal_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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

        if len(segregates) != 4:
            continue
        segregates = order_points(segregates)

        if len(segregates) != 4:
            continue



        corner_list = []

        for i in segregates:
            corner_list.append((i[0], i[1]))

        assert len(corner_list) == 4
        # print(corner_list)
        rand1= random.randrange(25,35)
        rand2 = random.randrange(25,35)
        rand3 = random.randrange(25,35)
        rand4 = random.randrange(25,35)
        rand5 = random.randrange(25,35)
        rand6 = random.randrange(25,35)
        rand7 = random.randrange(25,35)
        rand8 = random.randrange(25,35)

        h, status = cv2.findHomography(
            np.array(corner_list), np.array([[0+rand1, 0+rand2], [0+rand3, 224-rand4], [224-rand5, 224-rand6], [224-rand7, 0+rand8]]))
        height, width, channels = img.shape
        im1Reg = cv2.warpPerspective(img, h, (224, 224))


        label = response_id[int(corner_list[0][1]), int(corner_list[0][0]), 0]

        cv2.imwrite(os.path.join(args.out_folder, 'ssimg',
                                 filename[:-4] + "_" + str(index) + '.jpg'), im1Reg)

        with open(os.path.join(args.out_folder, 'ssimg',filename[:-4] + "_" + str(index) + '.txt'), "w") as text_file:
            print(f"{label}", file=text_file)

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
            img_out, response_1, response_2, response_3 ,response_id, corners_collection = overlayer(img)

            cv2.imwrite(os.path.join(args.out_folder, 'img',
                                     filename[:-4] + "_" + str(j) + '.jpg'), img_out)
            cv2.imwrite(os.path.join(args.out_folder, 'mask',
                                     filename[:-4] + "_" + str(j) + '_5.png'), response_1)
            for k in range(1):
                cv2.imwrite(os.path.join(args.out_folder, 'mask',  filename[:-4] + "_" + str(
                    j) + '_'+str(k) + '.png'), response_2[:, :, k])
            with open(os.path.join(args.out_folder, 'img',  filename[:-4] + "_" + str(j)   + '.pkl'), 'wb') as f:
                pickle.dump(corners_collection, f)


def run_multiprocessing(func, file_list, overlayer, args, n_processors):
    parameters = ((file, overlayer, args) for file in file_list)
    with Pool(processes=n_processors) as pool:
        return pool.starmap(func, parameters)


def app():
    parser = argparse.ArgumentParser(description='April tag image Generator.')
    parser.add_argument(
        '--root',
        type=str,
        default='/raid/apant_ma/AprilTag-Detection/AprilTag_Detection/DatasetGeneration/.',
        help='Directory to all standard April tag images.')
    parser.add_argument(
        '--img_folder',
        type=str,
        default='/raid/apant_ma/AprilTag-Detection/AprilTag_Detection/DatasetGeneration/../../dataset/',
        help='Folder which contains background images')
    parser.add_argument(
        '--out_folder',
        type=str,
        default='/raid/apant_ma/AprilTag-Detection/AprilTag_Detection/DatasetGeneration/./out',
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
    os.makedirs(os.path.join(args.out_folder, 'ssimg'), exist_ok=True)

    generator = AprilTagGenerator(root=args.root,
                                  family=args.family,
                                  size=args.size,
                                  rx_lim_deg=(-70, 70),
                                  ry_lim_deg=(-70, 70),
                                  rz_lim_deg=(-180, 180),
                                  scalex_lim=(1.0/128, 1.0/8),
                                  scaley_lim=(1.0/128, 1.0/8),
                                  )

    print(len(generator))
    overlayer = backgroundOverlayer(generator, args.mx_tags)
    directory = os.fsencode(args.img_folder)
    i = 0

    n_processors = 16

    mx_files = 4000

    file_list = sorted(list(os.listdir(directory))[2*mx_files:4*mx_files])

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
