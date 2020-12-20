import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.cm as cm


class AprilTagFormatter(object):
    """
    Formats the AprilTag coordinates for compatibity with calibrateCamera
    """
    def __call__(self, corners, ids, board):
        boardPoints = board.objPoints
        boardIds = board.ids

        boardIds = np.squeeze(boardIds, axis = 1)

        boardsIdsMapping = dict(zip(boardIds, boardPoints))

        objPoints = [np.array([boardsIdsMapping[ 300 + 324 - id.item()]  for id in single_board_ids if id.item() in boardsIdsMapping.keys()], dtype =np.float32).reshape(-1, 3) for single_board_ids in ids if len(single_board_ids)]

        return objPoints

class AprilTagPlotter(object):
    """
    Plots the corners of AprilTags in color coded form
    """
    def __call__(self, corners, ids):

        colors = itertools.cycle(["r", "b", "g", "black"])


        for img_corners, img_ids in zip(corners, ids):

            img_corners = np.array(img_corners)[..., :2].reshape(-1,  2)


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
