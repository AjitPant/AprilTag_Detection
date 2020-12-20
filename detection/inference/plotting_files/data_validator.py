import numpy as np
import pickle
import cv2
import glob
import os
import argparse
from multiprocessing import Process, Manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools




def process(args):
    corners, ids, counter = pickle.load( open( args.pkl_path, "rb" ) )[:3]



    colors = itertools.cycle(["r", "b", "g", "black"])

    for img_corners, img_ids, img_counter in zip(corners, ids, counter):
        print(img_ids)
        print("hi")
        print(np.array(img_corners).shape)
        img_corners = np.array(img_corners).reshape(-1,4,  2)
        img_corners[:,[0, 1,2, 3], :] = img_corners[:, [2,3,0,1], :]
        img_corners = img_corners.reshape(-1, 2)


        for corner in range(4):
            series = img_corners[corner::4]
            x, y = series.T
            y = -y
            plt.scatter(x,y, color=next(colors))

        img_corners = np.array(img_corners).reshape(-1,4,  2)
        for i in range(len(img_corners)):
            x = np.mean(img_corners[i,:,0])
            y = -np.mean(img_corners[i,:,1])
            plt.annotate(str(img_ids[i]),xy=(x,y))

        plt.show()






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pkl_path',
        type=str,
        required=True,
        help="Pickle file containing the corners list")
    args = parser.parse_args()
    process(args)


if __name__ == '__main__':
    main()
