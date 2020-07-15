
import argparse
import cv2
import numpy as np
from apriltag_images import TAG36h11, AprilTagImages
from apriltag_generator import AprilTagGenerator
from backgound_overlayer import backgroundOverlayer

import os
import random

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
        default=10,
        help='Maximum number of tags to generate in an image')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_folder, 'img'), exist_ok=True)
    os.makedirs(os.path.join(args.out_folder, 'mask'), exist_ok=True)




    generator = AprilTagGenerator(root=args.root,
                                  family=args.family,
                                  size=args.size,
                                  rx_lim_deg=(-50, 50),
                                  ry_lim_deg=(-50, 50),
                                  rz_lim_deg=(-180, 180),
                                  scalex_lim=(0.5, 2),
                                  scaley_lim=(0.5, 2),
                                  )

    print(len(generator))
    overlayer = backgroundOverlayer(generator, args.mx_tags)
    directory = os.fsencode(args.img_folder)
    i = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            path = (os.path.join(args.img_folder, filename))
            img = cv2.imread(path)
            if img is None:
                print("Failed to load the {}. Make sure it exists.", path)
                exit()

            img = cv2.resize(img, (256, 256))
            img_out, response_1, response_2 = overlayer(img)

            cv2.imwrite(os.path.join(args.out_folder, 'img', filename[:-4] + '.jpg'), img_out)
            cv2.imwrite(os.path.join(args.out_folder, 'mask',  filename[:-4]  + '_1.png'), response_1)
            cv2.imwrite(os.path.join(args.out_folder, 'mask',  filename[:-4]  + '_2.png'), response_2)

        #Usually ~20000 images give good result
        if(i==40000):
            break
        i+=1
        print(i)

if __name__ == "__main__":
    app()
