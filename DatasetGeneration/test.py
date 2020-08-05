import argparse
import cv2

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
        '--family',
        type=str,
        default=TAG36h11,
        help='April tag family.')
    parser.add_argument(
        '--perspective_percentage',
        type=float,
        default=.3,
        help='Perspective percentage for homography augmentation.')
    parser.add_argument(
        '--size',
        type=int,
        default=128,
        help='Size of April tag images in pixels.')
    args = parser.parse_args()

    generator = AprilTagGenerator(root=args.root,
                                  family=args.family,
                                  size=args.size,
                                  rx_lim_deg=(-80, 80),
                                  ry_lim_deg=(-80, 80),
                                  rz_lim_deg=(-180, 180),
                                  scalex_lim=(1,1),
                                  scaley_lim=(1,1))
    print(len(generator))

    for i in range(len(generator)):
        result = generator[i]

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.imshow("image", result["image"])

        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.imshow("mask", result["mask"])

        response = result["response"][:, :, :3]
        response[:, :, 0] = response[:, :, 0] + result["response"][:, :, 3]
        response[:, :, 1] = response[:, :, 1] + result["response"][:, :, 3]
        cv2.namedWindow("response", cv2.WINDOW_NORMAL)
        cv2.imshow("response", response)

        garbage = result["response"][:, :, 4]
        cv2.namedWindow("garbage", cv2.WINDOW_NORMAL)
        cv2.imshow("garbage", garbage)

        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    app()
