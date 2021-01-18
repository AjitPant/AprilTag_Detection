import cv2
import numpy as np
from euler import euler_angles_to_rotation_matrix


def deg_to_rad(x):
    return x * np.pi / 180.0


class Rotation3DAugmentation(object):
    """
    Augment images by 3D rotating image plane around the image center.
    """
    """ 3D Rotation transformation class for image"""

    def __init__(self,
                 rx_lim_deg=(0., 0.),
                 ry_lim_deg=(0., 0.),
                 rz_lim_deg=(0., 0.),
                 prob=0.0,
                 debug=False):
        '''
        :param rx_lim_deg: tuple(float, float) range of rotation around x axis in degree
        :param ry_lim_deg: tuple(float, float) range of rotation around y axis in degree
        :param rz_lim_deg: tuple(float, float) range of rotation around z axis in degree
        :param debug: If set, display processing images
        '''
        assert rx_lim_deg[0] <= rx_lim_deg[1], \
            'rx_lim[0]:{} < rx_lim:{}'.format(rx_lim_deg[0], rx_lim_deg[1])
        self.rx_lim = (deg_to_rad(rx_lim_deg[0]), deg_to_rad(rx_lim_deg[1]))
        assert ry_lim_deg[0] <= ry_lim_deg[1], \
            'ry_lim[0]:{} < ry_lim:{}'.format(ry_lim_deg[0], ry_lim_deg[1])
        self.ry_lim = (deg_to_rad(ry_lim_deg[0]), deg_to_rad(ry_lim_deg[1]))
        assert rz_lim_deg[0] <= rz_lim_deg[1], \
            'rz_lim[0]:{} < rz_lim:{}'.format(rz_lim_deg[0], rz_lim_deg[1])
        self.rz_lim = (deg_to_rad(rz_lim_deg[0]), deg_to_rad(rz_lim_deg[1]))

        assert 0. <= prob <= 1., 'prob ({}) must be in the range of [0, 1]'.format(
            prob)
        self.prob = prob
        self.debug = debug

    def __call__(self, src_image, src_corners):
        if np.random.uniform(0, 1, 1)[0] >= self.prob:
            return src_image, src_corners

        height, width = src_image.shape[:2]
        rx = 0

        rx = np.random.uniform(self.rx_lim[0], self.rx_lim[1], 1)[0]
        ry = np.random.uniform(self.ry_lim[0], self.ry_lim[1], 1)[0]
        rz = np.random.uniform(self.rz_lim[0], self.rz_lim[1], 1)[0]

        if np.random.uniform(0, 1, 1)>0.5:
            rx = np.random.uniform(self.rx_lim[0], self.rx_lim[1], 1)[0]
        else:
            rx = np.random.uniform(-self.rx_lim[1], -self.rx_lim[0], 1)[0]

        if np.random.uniform(0, 1, 1)>0.5:
            ry = np.random.uniform(self.ry_lim[0], self.ry_lim[1], 1)[0]
        else:
            ry = np.random.uniform(-self.ry_lim[1], -self.ry_lim[0], 1)[0]


        src_R_dst = euler_angles_to_rotation_matrix(rx, ry, rz).astype(np.float32)

        camera_matrix = np.eye(4, dtype=np.float32)
        camera_matrix[0, 2] = 0.5
        camera_matrix[1, 2] = 0.5 * height / width

        camera_T_plane = np.eye(4, dtype=np.float32)
        camera_T_plane[2, 3] = 1.0



        dst_corners = np.zeros((2, 4 ,2), dtype = np.float32)
        for ind in range(2):
            src_pts_img = np.ones( (src_corners.shape[1], 4), dtype=np.float32)
            src_pts_img[0, :] = src_corners[ind, :, 0]
            src_pts_img[1, :] = src_corners[ind, :, 1]

            src_pts_cam = np.matmul(np.linalg.inv(camera_matrix), src_pts_img)
            src_pts_pln = np.matmul(np.linalg.inv(camera_T_plane), src_pts_cam)
            dst_pts_pln = np.matmul(np.linalg.inv(src_R_dst), src_pts_pln)
            dst_pts_cam = np.matmul(camera_T_plane, dst_pts_pln)
            dst_pts_img = np.matmul(camera_matrix, dst_pts_cam)

            dst_pts_img[0] = dst_pts_img[0] / dst_pts_img[2]
            dst_pts_img[1] = dst_pts_img[1] / dst_pts_img[2]
            dst_corners[ind] = dst_pts_img[:2].T

        dst_H_src = cv2.getPerspectiveTransform(src_corners[0] * width, dst_corners[0] * width)
        dst_image = cv2.warpPerspective(src_image, dst_H_src, (width, height))

        return dst_image,  dst_corners



class ScaleAugmentation(object):
    """
    Augment images by scaling the height and width of the image.
    """
    """ Scale transformation class for image"""

    def __init__(self,
                 scalex_lim=(1., 1.),
                 scaley_lim=(1., 1.),
                 prob=0.0,
                 debug=False):
        '''
        :param scalex_lim_deg: tuple(float, float) range of ratio of final tag's \
                               width to original tag's width
        :param scaley_lim_deg: tuple(float, float) range of ratio of final tag's \
                               height to original tag's height
        :param debug: If set, display processing images
        '''
        assert scalex_lim[0] <= scalex_lim[1], \
            'scalex_lim[0]:{} < scalex_lim:{}'.format(scalex_lim[0], scalex_lim[1])
        self.scalex_lim = scalex_lim


        assert scaley_lim[0] <= scaley_lim[1], \
            'scaley_lim[0]:{} < scaley_lim:{}'.format(scaley_lim[0], scaley_lim[1])
        self.scaley_lim = scaley_lim

        assert 0. <= prob <= 1., 'prob ({}) must be in the range of [0, 1]'.format(
            prob)
        self.prob = prob
        self.debug = debug

    def __call__(self, src_image, src_corners):
        if np.random.uniform(0, 1, 1)[0] >= self.prob:
            return src_image, src_corners




        scalex = np.random.uniform(self.scalex_lim[0], self.scalex_lim[1], 1)[0]
        scaley = np.random.uniform(self.scaley_lim[0], self.scaley_lim[1], 1)[0]

        if np.random.uniform(0, 1, 1)[0] >= self.prob:
            scalex = np.random.uniform(self.scalex_lim[0], 0.5, 1)[0]
            scaley = np.random.uniform(self.scaley_lim[0], 0.5, 1)[0]
# GAussian

        dst_image = cv2.resize(src_image, interpolation = cv2.INTER_AREA,dsize = None, fx= scalex, fy = scaley)
        dst_corners = src_corners

        return dst_image,  dst_corners
