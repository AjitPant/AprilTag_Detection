import cv2
import numpy as np
import os

TAG36h11 = 'tag36h11'


class AprilTagImages(object):
    def __init__(self, root, family, size):
        """
        April Tag images.
        :param root: Directory of April tag images containing different standards (i.e. 36h11).
        :param family: April tag family class.
        :param size: Size of images in pixels.
        """
        assert size >= 10, 'April tag size must be equal or greater than 10.'
        self.size = size
        self.family = family
        # Path to mosaic image of April Tag images.
        self.path = os.path.join(root, os.path.join(family, 'mosaic.png'))
        self.images = self.extract_images()

    def __len__(self):
        """
        :return: Number of April tag images.
        """
        return len(self.images)

    def extract_images(self):
        """
        Extract April tag images from mosaic image containing all tags.
        :return:
        """
        assert os.path.exists(self.path), "{} does not exist!".format(self.path)
        mosaic_image = cv2.imread(self.path)[:, :, 0]
        nsquares_y = mosaic_image.shape[0] - np.count_nonzero(np.sum(mosaic_image, axis=1, dtype=np.int)) + 1
        nsquares_x = mosaic_image.shape[1] - np.count_nonzero(np.sum(mosaic_image, axis=0, dtype=np.int)) + 1

        if self.family == TAG36h11:
            step = 10
        else:
            assert False, 'Unknown April tag family!' + self.family

        images = list()
        for i in range(nsquares_y):
            for j in range(nsquares_x):
                row = i * (step + 1)
                col = j * (step + 1)
                image = mosaic_image[row:row + step, col:col + step]
                if np.sum(image) == 0:
                    break
                scaled_image = cv2.resize(
                    image,
                    dsize=(self.size, self.size),
                    interpolation=cv2.INTER_NEAREST)
                images.append(scaled_image)
        return images

    def corners(self):

        if self.family == TAG36h11:
            lo = 0.1
            hi = 0.9
        else:
            assert False, 'Unknown April tag family!' + self.family
        return np.array([[[lo, lo], [hi, lo], [hi, hi], [lo, hi]],
                         [[0.05, 0.05], [1-0.05, 0.05], [1-0.05, 1-0.05], [0.05, 1-0.05]]], dtype=np.float32)

    def image(self, idx):
                    assert idx < len(self.images), 'Not a valid index.'
                    return self.images[idx]
