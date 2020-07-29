import argparse
import cv2
import os
import numpy as np
from apriltag_images import TAG36h11, AprilTagImages
from apriltag_generator import AprilTagGenerator


def app():
    parser = argparse.ArgumentParser(description='April tag image Generator.')
    parser.add_argument(
        '--root',
        type=str,
        default=None,
        help='Directory to all standard April tag images.')
    parser.add_argument(
        '--out_folder',
        type=str,
        default='./labels/out',
        help='Out folder for masks.')
    parser.add_argument(
        '--in_folder',
        type=str,
        default='./labels/in',
        help='Input folder for images.')
    args = parser.parse_args()



    os.makedirs(args.out_folder, exist_ok=True)

    directory = os.fsencode(args.in_folder)
    i = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".bmp"):
            path = (os.path.join(args.in_folder, filename))
            img = cv2.imread(path)
            if img is None:
                print("Failed to load the {}. Make sure it exists.", path)
                exit()

            img = cv2.resize(img, (256, 256))

            mask = np.zeros(img.shape[:2])

            #Load the dictionary that was used to generate the markers.
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)

            # Initialize the detector parameters using default values
            parameters =  cv2.aruco.DetectorParameters_create()

            # Detect the markers in the image
            markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

            padding = 1;
            for corners in markerCorners:
                corners = corners[0]
                for ind, corner in enumerate(corners):
                    row, col = corner
                    for r in range( max(0, int(row)-padding), min(img.shape[0], int(row)+padding)):
                        for c in range( max(0, int(col)-padding), min(img.shape[1], int(col)+padding)):
                            mask[r][c] = ind+1

            for corners in rejectedCandidates:
                corners = corners[0]
                for ind, corner in enumerate(corners):
                    row, col = corner
                    for r in range( max(0, int(row)-padding), min(img.shape[0], int(row)+padding)):
                        for c in range( max(0, int(col)-padding), min(img.shape[1], int(col)+padding)):
                            mask[r][c] = ind+1
            cv2.imwrite(os.path.join(args.out_folder, filename[:-4] + '.png'), mask)
            # cv2.imwrite(os.path.join(args.out_folder, 'mask',  filename[:-4]  + '_1.png'), response_1)
            # cv2.imwrite(os.path.join(args.out_folder, 'mask',  filename[:-4]  + '_2.png'), response_2)





if __name__ == "__main__":
    app()
