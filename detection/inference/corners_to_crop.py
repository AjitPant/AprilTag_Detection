import cv2
import numpy as np
from scipy import stats


import torch


def nonzero_mode(arr):
    return stats.mode(arr[np.nonzero(arr)]).mode
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")



def crop_to_corners(net, img, mask, device):
    # Finds the corners in the image and then crop them
    segregates = []

    mask_segmentation = mask[0]*255
    mask_corners = mask[1]

    # print(mask_segmentation.shape)
    # print(mask_corners.shape)
    # print(mask_segmentation)
    # print(mask_corners)

    kernel = np.ones((5,5), np.uint8)
    # mask_segmentation = cv2.erode(mask_segmentation, kernel, iterations=1)
    # mask_segmentation = cv2.dilate(mask_segmentation, kernel, iterations=1)
    # cv2.namedWindow('mask_segmentation', cv2.WINDOW_NORMAL)
    # cv2.imshow(     "mask_segmentation", mask_segmentation)


    cv2.namedWindow('mask_garbage_0', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask_garbage_1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask_garbage_2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask_garbage_3', cv2.WINDOW_NORMAL)
    cv2.namedWindow('mask_garbage_4', cv2.WINDOW_NORMAL)

    for i in range(mask_corners.shape[0]):
        cv2.imshow("mask_garbage_"+str(i), np.clip(mask_corners[i,:,:],0,255))
        print(mask_corners[i,:,:].max())

    mask_real_corners = np.zeros(mask_corners.shape[1:],dtype = np.uint8)
    print(mask_corners.max())
    # for i in range(4):
    #     mask_real_corners +=mask_corners[i]*(i+1)
    mask_real_corners = mask_corners
    mask_real_corners = 4- np.argmax(mask_corners[:5], axis = 0).astype(np.uint8)

    cv2.namedWindow('mask_garbage', cv2.WINDOW_NORMAL)
    cv2.imshow("mask_garbage", mask_real_corners*60)
    cv2.waitKey(0)

    # return
    # Divide into four masks and then compute for drawing only
    # mask_corners_list = []
    # for i in range(1, 5):
    #     mask_corners_list.append(mask_corners == i)


    # cv2.imshow("mask_garbage", mask_corners*60)


    kernel = np.ones((5,5),np.uint8)
    # mask_segmentation = cv2.dilate(mask_segmentation, kernel)

    contours, _= cv2.findContours(mask_segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    temp_img = img.copy();
    coords = np.argwhere(mask_corners > 0)

    cv2.drawContours(temp_img, contours, -1, (0,255,0), 3)

    cv2.namedWindow('contours_img', cv2.WINDOW_NORMAL)
    cv2.imshow("contours_img", temp_img)
    cv2.waitKey(0)



    for ind in range(len(contours)):
        segregates = []
        internal_mask = np.zeros(mask_real_corners.shape, dtype = np.uint8)

        cv2.drawContours(internal_mask, contours, ind, 255, -1)

        cv2.namedWindow('internal_mask', cv2.WINDOW_NORMAL)
        cv2.imshow("internal_mask", internal_mask)
        cv2.waitKey(0)
        internal_mask = cv2.bitwise_and(internal_mask, mask_real_corners)

        # inner_contours, _= cv2.findContours(internal_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # for inner_ind in range(len(internal_contours)):
        #     internal_internal_mask = np.zeros(mask_real_corners.shape, dtype = np.uint8)
        #     cv2.drawContours(internal_internal_mask, internal_contours, inner_ind, 255, -1)
        #     internal_internal_mask = cv2.bitwise_and(internal_internal_mask, mask_real_corners)

        #     mode = map(nonzero_mode, internal_internal_mask)
        #     #find the center of contours
        #     M = cv2.moments(c)
        #     cX = int(M["m10"] / M["m00"])
        #     cY = int(M["m01"] / M["m00"])
        #     segregates.append([ cX, cY, M])

        # segregates = order_points(segregates)
        # if len(segregates)!=4:
        #     continue;
        # #     # for i in range(1, 5):
        # #     #     segregateV.append(np.median((np.argwhere(internal_mask == i)), axis=0))

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
