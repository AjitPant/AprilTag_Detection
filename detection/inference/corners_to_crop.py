import cv2
import numpy as np


import torch


def crop_to_corners(net, img, mask, device):
    # Finds the corners in the image and then crop them
    segregates = []

    mask_segmentation = mask[0]*255
    mask_corners = mask[1]

    print(mask_segmentation.shape)
    print(mask_corners.shape)
    print(mask_segmentation)
    print(mask_corners)

    kernel = np.ones((5,5), np.uint8)
    # mask_segmentation = cv2.erode(mask_segmentation, kernel, iterations=1)
    # mask_segmentation = cv2.dilate(mask_segmentation, kernel, iterations=1)
    cv2.namedWindow('mask_segmentation', cv2.WINDOW_NORMAL)
    cv2.imshow(     "mask_segmentation", mask_segmentation)
    cv2.namedWindow('mask_garbage', cv2.WINDOW_NORMAL)


    # Divide into four masks and then compute for drawing only
    mask_corners_list = []
    for i in range(1, 5):
        mask_corners_list.append(mask_corners == i)
        

    cv2.imshow("mask_garbage", mask_corners*60)


    kernel = np.ones((5,5),np.uint8)
    contours, _= cv2.findContours(mask_segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    temp_img = img.copy();
    coords = np.argwhere(mask_corners > 0)

    cv2.drawContours(temp_img, contours, -1, (0,255,0), 3)

    cv2.namedWindow('contours_img', cv2.WINDOW_NORMAL)
    cv2.imshow("contours_img", temp_img)
    cv2.waitKey(0)



    for ind in range(len(contours)):
        segregates = []
        internal_mask = np.zeros(mask_corners.shape, dtype = np.uint8)

        cv2.drawContours(internal_mask, contours, ind, 255, -1)

        cv2.namedWindow('internal_mask', cv2.WINDOW_NORMAL)
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
        cv2.namedWindow('a', cv2.WINDOW_NORMAL)
        cv2.imshow('a', im1Reg)
        cv2.waitKey(0)

        # border= ((0, 0), im1Reg.shape);
        # color = 0
        # thickness = 1;

        # cv2.rectangle(im1Reg, (0,0),(224, 224), color, thickness);

        inp = torch.tensor(im1Reg).to(device).float()/255
        print(inp)
        # label = net(inp.T.unsqueeze(0)).argmax()
        # print(label)





if __name__ == '__main__':
    mask = [cv2.imread('./testing/mask_2.png', cv2.IMREAD_GRAYSCALE), cv2.imread('./testing/mask_1.png', cv2.IMREAD_GRAYSCALE),]
    img = cv2.imread('./testing/img.jpg')
    crop_to_corners(img, mask)
