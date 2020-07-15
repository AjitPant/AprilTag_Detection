import cv2
import numpy as np
n_classes = 5


def crop_to_corners(img, mask):
    # Finds the corners in the image and then crop them
    segregates = []

    mask_segmentation = mask[0]*255
    mask_corners = mask[1]

    cv2.imshow("mask_segmentation", mask_segmentation)
    cv2.imshow("mask_garbage", mask_corners*30)
    cv2.waitKey(0)

    kernel = np.ones((5,5),np.uint8)
    contours, _= cv2.findContours(cv2.morphologyEx(mask_segmentation, cv2.MORPH_CLOSE, kernel), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    coords = np.argwhere(mask_corners > 0)

    cv2.drawContours(img, contours, -1, (0,255,0), 3)


    cv2.imshow("contours_img", img)
    cv2.waitKey(0)



    for ind in range(len(contours)):
        segregates = []
        internal_mask = np.zeros(mask_corners.shape, dtype = np.uint8)

        cv2.drawContours(internal_mask, contours, ind, 255, -1)

        cv2.imshow("internal_mask", internal_mask)
        cv2.waitKey(0)
        internal_mask = cv2.bitwise_and(internal_mask, mask_corners)

        for i in range(1, 5):
            segregates.append(np.median((np.argwhere(internal_mask == i)), axis=0))


        corner_list = []
        print(segregates)
        for i in segregates:
            corner_list.append((i[1], i[0]))

        assert len(corner_list) == 4
        print(corner_list)

        h, status = cv2.findHomography(
                    np.array(corner_list), np.array([[0, 0], [0, 224], [224, 224], [224, 0]]))
        height, width, channels = img.shape
        im1Reg = cv2.warpPerspective(img, h, (224, 224))
        cv2.imshow('a', im1Reg)
        cv2.waitKey(0)


if __name__ == '__main__':
    mask = [cv2.imread('./testing/mask_2.png', cv2.IMREAD_GRAYSCALE), cv2.imread('./testing/mask_1.png', cv2.IMREAD_GRAYSCALE),]
    img = cv2.imread('./testing/img.jpg')
    crop_to_corners(img, mask)
