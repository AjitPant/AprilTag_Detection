
import argparse
import cv2
import numpy as np
from apriltag_images import TAG36h11, AprilTagImages
from apriltag_generator import AprilTagGenerator
from backgound_overlayer import backgroundOverlayer

import multiprocessing
from multiprocessing import Pool
from multiprocessing import freeze_support
import itertools

import copy
import os
import random
import time

def augment_and_save(file, overlayer, args):
    filename = os.fsdecode(file)
    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".bmp"):
        path = (os.path.join(args.img_folder, filename))
        for j in range(1):
            img = cv2.imread(path)
            if img is None:
                print("Failed to load the {}. Make sure it exists.", path)
                exit()

            img = cv2.resize(img, (512, 512))
            img_out, response_1, response_2 = overlayer(img)

            cv2.imwrite(os.path.join(args.out_folder, 'img', filename[:-4] + "_" + str(j) + '.jpg'), img_out)
            cv2.imwrite(os.path.join(args.out_folder, 'mask',  filename[:-4] + "_" + str(j) + '_5.png'), response_1)
            for k in range(5):
                cv2.imwrite(os.path.join(args.out_folder, 'mask',  filename[:-4] + "_" + str(j) + '_'+str(k)+ '.png'), response_2[:,:,k])


def run_multiprocessing(func, file_list, overlayer, args, n_processors):
    parameters = ((file, overlayer, args ) for file in file_list)
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
        default=64,
        help='Size of April tag images in pixels.')
    parser.add_argument(
        '--mx_tags',
        type=int,
        default=30,
        help='Maximum number of tags to generate in an image')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_folder, 'img'), exist_ok=True)
    os.makedirs(os.path.join(args.out_folder, 'mask'), exist_ok=True)


    # logger = multiprocessing.log_to_stderr()
    # logger.setLevel(multiprocessing.SUBDEBUG)
    generator = AprilTagGenerator(root=args.root,
                                  family=args.family,
                                  size=args.size,
                                  rx_lim_deg=(-30, 30),
                                  ry_lim_deg=(-30, 30),
                                  rz_lim_deg=(-180, 180),
                                  scalex_lim=(0.40, 4.0),
                                  scaley_lim=(0.40, 4.0),
                                  )

    print(len(generator))
    overlayer = backgroundOverlayer(generator, args.mx_tags)
    directory = os.fsencode(args.img_folder)
    i = 0

    n_processors = 10

    mx_files =100

    file_list = sorted(list(os.listdir(directory))[:mx_files])

    '''
    pass the task function, followed by the parameters to processors
    '''
    start = time.time()
    out = run_multiprocessing(augment_and_save, file_list, overlayer, args, n_processors)
    print("Mutiprocessing time: {}secs\n".format((time.time()-start)))



if __name__ == "__main__":
    freeze_support()   # required to use multiprocessing
    app()
