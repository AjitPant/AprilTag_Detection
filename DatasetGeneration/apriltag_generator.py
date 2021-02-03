import cv2
import numpy as np
from apriltag_images import AprilTagImages
from image_augmentation import Rotation3DAugmentation, ScaleAugmentation
import random

class AprilTagGenerator(object):
    """
    Generate synthetic April Tag images.
    """
    def __init__(self, root, family, size, rx_lim_deg, ry_lim_deg, rz_lim_deg,
                 scalex_lim, scaley_lim):
        """
        :param root: directory to April Tag images.
        :param family: April Tag family.
        :param size: size of generated images in pixels
        :param theta: Maximum rotation angle (in degrees) of normal vector of tag's plane.
        """
        self.size = size
        self.apriltags = AprilTagImages(root=root, family=family, size=size)
        self.rotate3d = Rotation3DAugmentation(rx_lim_deg=rx_lim_deg,
                                               ry_lim_deg=ry_lim_deg,
                                               rz_lim_deg=rz_lim_deg)
        self.scale = ScaleAugmentation(scalex_lim=scalex_lim,
                                       scaley_lim=scaley_lim)

    def __len__(self):
        """
        :return: Number of April Tags (i.e. 587 for TAG36h11 family).
        """
        return len(self.apriltags)

    def __getitem__(self, idx):
        assert idx < len(self.apriltags), 'Not a valid index.'
        pad_width = self.size // 2
        src_image, src_bytecode, src_familycode = self.apriltags.image(idx)
        src_image = np.pad(src_image, pad_width=pad_width, mode='constant', constant_values=255)
        src_width = src_image.shape[1]

        src_corners_uv = (self.apriltags.corners() * self.apriltags.size + pad_width) / src_width

        # Warped April Tag image and its normalized uv corners.
        dst_image,  dst_corners_uv = self.rotate3d(src_image, src_corners_uv)
        dst_image, dst_corners_uv = self.scale(dst_image, dst_corners_uv)

        dst_height, dst_width = dst_image.shape[:2]


        mask = np.zeros((dst_height, dst_width), dtype=np.uint8)

        polygon = (dst_corners_uv[1] * dst_image.shape[::-1]).astype(np.int32)
        cv2.fillPoly(mask, np.array([polygon]), 255)


        return dict(
            image=dst_image,
            bytecode = src_bytecode,
            familycode = src_familycode,
            mask=mask,
            response=self.get_response(dst_corners_uv[0], dst_width, dst_height)[0],
            response_in_use=self.get_response(dst_corners_uv[0], dst_width, dst_height)[1],
            corners_uv=dst_corners_uv[0]*np.array([dst_width, dst_height]))


    @staticmethod
    def get_response(corners, width, height):
        """
        Calculate response map of corners.
        :param corners: np.ndarray[N,2]
        :param width: int (in pixels)
        :param height: int (in pixels)
        :return: np.ndarray[N+1,H,W] where N is number of corners
        """
        num_feats = corners.shape[0]
        response = np.zeros((height, width, num_feats + 1), dtype=np.uint8)
        response[:, :, -1] = 255

        response_in_use = np.zeros((height, width, num_feats + 1), dtype=np.uint8)
        response_in_use[:, :, -1] = 255

        d1 = 1
        d2 = 1
        for i in range(num_feats):
            assert 0.0 <= corners[i, 0] < 1.0, 'corner x outside of image border!'
            assert 0.0 <= corners[i, 1] < 1.0, 'corner y outside of image border!'
            x = int(round(corners[i, 0] * width))
            y = int(round(corners[i, 1] * height))



            response[max(0, y - d2):min(height, y + d2 + 1), max(0, x - d2):min(width, x + d2 + 1), 0] = 255
            response[max(0, y - d2):min(height, y + d2 + 1), max(0, x - d2):min(width, x + d2 + 1), -1] = 0

            response_in_use[max(0, y - d2):min(height, y + d2 + 1), max(0, x - d2):min(width, x + d2 + 1), i] = 255
            response_in_use[max(0, y - d2):min(height, y + d2 + 1), max(0, x - d2):min(width, x + d2 + 1), -1] = 0
            # print(np.random.multivariate_normal((x, y), [[1,0],[0,1]], size = (2,2)))
        return response, response_in_use
