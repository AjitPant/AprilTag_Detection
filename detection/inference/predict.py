import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
import dataset_classifier

from Unet import Unet
from classifier import Resnet
from corners_to_crop import crop_to_corners
from dataset import DirDataset

from torchvision import datasets, models, transforms


from scipy.spatial import distance as dist
from scipy import stats


import torch



import operator
import math
from functools import reduce



def nonzero_mode(arr):
    return stats.mode(arr[np.nonzero(arr)]).mode

from scipy.spatial import distance as dist
import numpy as np
import cv2
def order_points(pts):
	coords = pts
	center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
	out = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
	return np.array(out, dtype="float32")


def reduce_to_tags(img, response_1, response_2, filename, hparams):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(hparams.id_net)
    net = Resnet.load_from_checkpoint(hparams.id_net)
    net.freeze()
    net.to(device)
    net.eval()

    mask_segmentation = response_1
    mask_corners = response_2
    segregates = []

    mask_corners =  (mask_corners>0.5).astype(np.uint8)
    print(mask_corners.shape)


    cv2.namedWindow('mask_segmentation', cv2.WINDOW_NORMAL)
    cv2.imshow("mask_segmentation", mask_segmentation*255)

    # cv2.namedWindow('mask_garbage_0', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('mask_garbage_1', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('mask_garbage_2', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('mask_garbage_3', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('mask_garbage_4', cv2.WINDOW_NORMAL)

    mask_real_corners = np.zeros(mask_corners.shape[1:], dtype=np.uint8)
    # print(mask_corners.max())
    # for i in range(4):
    #     mask_real_corners +=mask_corners[i]*(i+1)
    mask_real_corners = (mask_corners!=0).astype(np.uint8).squeeze(0)
    print(mask_real_corners.shape)
    # mask_real_corners = 4- np.argmax(mask_corners, axis = 0).astype(np.uint8)
    # print(mask_real_corners)

    cv2.namedWindow('mask_garbage', cv2.WINDOW_NORMAL)
    cv2.imshow("mask_garbage", mask_real_corners.astype(np.float32)*60)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(
        mask_segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    temp_img = img.copy()
    coords = np.argwhere(mask_corners > 0)

    cv2.drawContours(temp_img, contours, -1, (0, 255, 0), 3)

    cv2.namedWindow('contours_img', cv2.WINDOW_NORMAL)
    cv2.imshow("contours_img", temp_img)
    cv2.waitKey(0)
    index = 0

    for ind in range(len(contours)):
        segregates = []
        internal_mask = np.zeros(mask_real_corners.shape, dtype=np.uint8)

        cv2.drawContours(internal_mask, contours, ind, 255, -1)

        cv2.namedWindow('internal_mask', cv2.WINDOW_NORMAL)
        cv2.imshow("internal_mask", internal_mask)
        cv2.waitKey(0)
        # print("hello")
        internal_mask = cv2.bitwise_and(
            internal_mask, mask_real_corners.astype(np.uint8))
        # print("heello")

        internal_contours, _ = cv2.findContours(
            internal_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print("heeello")
        # print(len(internal_contours))
        for inner_ind in range(len(internal_contours)):
            internal_internal_mask = np.zeros(
                mask_real_corners.shape, dtype=np.uint8)
            cv2.drawContours(internal_internal_mask,
                             internal_contours, inner_ind, 255, -1)
            internal_internal_mask = cv2.bitwise_and(
                internal_internal_mask, mask_real_corners.astype(np.uint8))

            mode = map(nonzero_mode, internal_internal_mask)
            #find the center of contours
            M = cv2.moments(internal_contours[inner_ind])
            cX = int(M["m10"] / (M["m00"]+1e-2))
            cY = int(M["m01"] / (M["m00"]+1e-2))
            segregates.append([cX, cY])
        print(segregates)
        if len(segregates) != 4:
            continue
        segregates = order_points(segregates)
        # print(len(segregates))
        if len(segregates) != 4:
            continue

        corner_list = []
        # print(segregates)
        for i in segregates:
            corner_list.append((i[0], i[1]))

        assert len(corner_list) == 4
        # print(corner_list)
        pad = 0
        h, status = cv2.findHomography(
            np.array(corner_list), np.array([[pad, pad], [pad, 224-pad],[224-pad, 224-pad],  [224-pad, 0+pad]]))
        height, width, channels = img.shape
        im1Reg = cv2.warpPerspective(img, h, (224, 224))
        cv2.namedWindow('unrotated_tag', cv2.WINDOW_NORMAL)
        cv2.imshow('unrotated_tag', im1Reg)
        cv2.waitKey(0)

        im2Reg = Image.fromarray(im1Reg)

        ds = dataset_classifier.DirDataset('', '')
        im2Reg = (ds.preprocess(im2Reg))
        out = net(im2Reg.unsqueeze(0).to(device))
        print(out)
        rotation = (np.argmax(out.squeeze(0).cpu())*-90).item()
        print(rotation)

        (h, w) = im1Reg.shape[:2]

        # calculate the center of the image
        center = (w / 2, h / 2)
        scale = 1

        M = cv2.getRotationMatrix2D(center, rotation, scale)
        im3Reg = cv2.warpAffine(im1Reg, M, (h, w))

        cv2.namedWindow('rotated_tag', cv2.WINDOW_NORMAL)
        cv2.imshow('rotated_tag', im3Reg)
        cv2.waitKey(0)


        # dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)

        # # Initialize the detector parameters using default values
        # parameters =  cv2.aruco.DetectorParameters_create()
        # # parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        # parameters.maxErroneousBitsInBorderRate = 0.8
        # parameters.errorCorrectionRate = 0.8
        # parameters.aprilTagMaxLineFitMse = 100

        # markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(im1Reg, dictionary, parameters=parameters)
        # print(im1Reg)
        # print(im1Reg.max())
        # print(im1Reg.min())

        # print(markerCorners)
        # print(rejectedCandidates)
        # for corners in markerCorners:
        #     corner = corners[0]
        #     print(corner)
        #     h, status = cv2.findHomography(
        #                 np.array(corner), np.array([[pad, pad], [224-pad, 0], [224-pad, 224-pad], [pad, 224-pad]]))
        #     height, width, channels = img.shape
        #     im1Regg = cv2.warpPerspective(im1Reg, h, (224, 224))
        #     cv2.namedWindow('classical_detection', cv2.WINDOW_NORMAL)
        #     cv2.imshow('classical_detection', im1Regg)
        #     cv2.waitKey(0)


        # for corners in rejectedCandidates:
        #     corner = corners[0]
        #     print(corners)
        #     h, status = cv2.findHomography(
        #                 np.array(corner), np.array([[pad, pad], [pad, 224-pad], [224-pad, 224-pad], [224-pad, 0+pad]]))
        #     height, width, channels = img.shape
        #     im1Regg = cv2.warpPerspective(im1Reg, h, (224, 224))
        #     cv2.namedWindow('classical_detection', cv2.WINDOW_NORMAL)
        #     cv2.imshow('classical_detection', im1Regg)
        #     cv2.waitKey(0)




        # print(corner_list)

        # cv2.imwrite(os.path.join(args.out_folder, 'simg',
        #                          filename[:-4] + "_" + str(index) + '.jpg'), im1Reg)


        # print(f"{mask_corners[int(corner_list[0][1])][int(corner_list[0][0])]}")
        # with open(os.path.join(args.out_folder, 'simg',filename[:-4] + "_" + str(index) + '.txt'), "w") as text_file:
        #     print(f"{mask_corners[int(corner_list[0][1])][int(corner_list[0][0])]}", file=text_file)

        index = index + 1



def predict(net, img, device='cpu', threshold=0.9):
    ds = DirDataset('', '')
    _img = (ds.preprocess(img))

    # cv2.imshow("predict", _img.cpu().numpy().transpose((2, 1, 0)))
    # cv2.waitKey(0)
    _img = _img.unsqueeze(0)
    _img = _img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        o = net(_img)

        _o = o[:, 1, :, :]
        o = o[:, :1,:,:]

        # probs = torch.nn.functional.softmax(o, dim=1)
        print(o.shape)
        probs = torch.sigmoid(o)
        print(o.max())
        print(probs.shape)
        probs = probs.squeeze(0)
        probs = probs.cpu()
        mask = probs.cpu().numpy()

        _probs = torch.sigmoid(_o)
        _probs = _probs.squeeze(0)
        _probs = _probs.cpu()
        _mask = _probs.squeeze().cpu().numpy()
    return (mask, _mask > threshold )


def mask_to_image(mask):
    return Image.fromarray(( mask ).astype(np.uint8))


def main(hparams):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Unet.load_from_checkpoint(hparams.checkpoint)
    # net.freeze()
    net.to(device)
    net.eval()

    #Load the identification network
    # model_ft = models.resnet18(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model_ft.fc = nn.Linear(num_ftrs, 587)

    # identification_net = model_ft.to(device)
    # identification_net.load_state_dict(torch.load(hparams.id_net))
    # identification_net.eval()


    im_size = 512
    img = Image.open(hparams.img)
    img = img.resize((im_size, im_size))

    mask, _mask = predict(net, img, device=device)
    # mask = 4 - mask.argmax(axis = 0);


    img = np.array(img)

    print(img.dtype)

    # mask = mask.astype(np.uint8)
    _mask = _mask.astype(np.uint8)
    # crop_to_corners(identification_net, img, [_mask, mask], device)
    reduce_to_tags(img, _mask, mask, '.', hparams)



if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help = False)
    parent_parser.add_argument('--checkpoint', required=True, help = "Network for segmentation")
    parent_parser.add_argument('--id_net', required=True, help = "Network for tag classification")
    parent_parser.add_argument('--img', required=True)

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
