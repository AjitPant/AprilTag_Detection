import pickle
from scipy.spatial import distance as dist
import time
import random
import os
import copy
import argparse
import cv2
import numpy as np
from apriltag_images import TAG36h11,TAG41h12, AprilTagImages
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
def reduce_to_tags(img,  corners_collection, bytecode_collection,familycode_collection, filename, args):
    pad = 80
    for j,(corners, bytecode, code) in enumerate(zip(corners_collection, bytecode_collection, familycode_collection)):
            h, status = cv2.findHomography(
                np.array(corners), np.array([[pad, pad], [pad, 224-pad], [224-pad, 224-pad], [224-pad, pad]]))
            height, width, channels = img.shape
            im1Reg = cv2.warpPerspective(img, h, (224, 224))


            cv2.imwrite(os.path.join(args.out_folder, 'ssimg',
                                     filename[:-4] + "_" + str(j) + '.jpg'), im1Reg)

            with open(os.path.join(args.out_folder, 'simg',
                                     filename[:-4] + "_" + str(j) + '.pkl'), "wb") as f:
                pickle.dump([bytecode, code], f)






def augment_and_save(file, overlayer, args):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".bmp"):
        path = (os.path.join(args.img_folder, filename))
        for j in range(1):
            img = cv2.imread(path)

            if img is None:
                print("Failed to load the {}. Make sure it exists.", path)
                exit()

            img = cv2.resize(img, (512*2*8, 512*2*8))
            img_out, response_1, response_2, response_3 ,response_id, corners_collection,bytecode_collection, familycode_collection= overlayer(img)

            img_out = cv2.resize(img_out, (1024, 1024), interpolation = cv2.INTER_AREA)
            response_1 = cv2.resize(response_1, (1024, 1024), interpolation = cv2.INTER_AREA)
            response_2 = cv2.resize(response_2, (1024, 1024), interpolation = cv2.INTER_AREA)



            corners_collection = [ [x/8 for x in y ]  for y in corners_collection]

#            reduce_to_tags(img_out, corners_collection,bytecode_collection,familycode_collection, filename, args)

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
        default='/raid/apant_ma/AprilTag-Detection/AprilTag_Detection/DatasetGeneration/dataset/36h11-11',
        help='Output folder which contains dataset')
    parser.add_argument(
        '--family',
        type=str,
        default=TAG36h11,
        help='April tag family.')
    parser.add_argument(
        '--size',
        type=int,
        default=320 * 4,
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
                                  rx_lim_deg=(-50, 50),
                                  ry_lim_deg=(-50, 50),
                                  rz_lim_deg=(-180, 180),
                                  scalex_lim=(1.0/128, 1.0),
                                  scaley_lim=(1.0/128, 1.0),
                                  )

    print(len(generator))
    overlayer = backgroundOverlayer(generator, args.mx_tags)
    directory = os.fsencode(args.img_folder)
    i = 0

    n_processors =40

    mx_files = 50

    file_list = sorted(list(os.listdir(directory))[0*mx_files:1*mx_files])

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
