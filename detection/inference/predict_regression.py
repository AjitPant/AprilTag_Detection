import pickle
import math
from matplotlib import pyplot as plt

from torch import nn
from argparse import ArgumentParser
import glob
import os
import cv2
import numpy as np

import torch

from PIL import Image
import dataset_classifier

from Unet_regression import Unet
from classifier import Resnet
from dataset import DirDataset
import operator
import math
from functools import reduce

cv_time_wait = 0

inf_ind = 0
class_ind = 0
def order_points(pts):
	coords = pts
	center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
	out = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
	return np.array(out, dtype="float32")

global_output_corners, global_output_id, global_output_cnter = [], [], []

def reduce_to_tags(net, img, response_1, response_2,  args):
    global inf_ind
    global class_ind
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detected = False


    mask_segmentation = response_1
    mask_segmentation[:] = 255
    mask_corners = response_2
    segregates = []


    mask_corners =  ((mask_corners* 255))

    mask_corners[mask_corners < 10] = 0

    print(np.unique(mask_corners))

    # kernel = np.ones((5,5),np.uint8)
    # mask_segmentation = cv2.erode(mask_segmentation,kernel,iterations = 2)


    cv2.namedWindow('mask_segmentation', cv2.WINDOW_NORMAL)
    cv2.imshow("mask_segmentation", mask_segmentation*255)
    cv2.waitKey(cv_time_wait)

    mask_real_corners = np.zeros(mask_corners.shape[1:], dtype=np.float32)

    mask_real_corners = (mask_corners)
    print(np.unique(mask_real_corners))

    cv2.namedWindow('mask_garbage', cv2.WINDOW_NORMAL)
    cv2.imshow("mask_garbage", mask_real_corners)
    cv2.waitKey(cv_time_wait)




    contours, _ = cv2.findContours(
        mask_segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    temp_img = img.copy()

    cv2.drawContours(temp_img, contours, -1, (0, 255, 0), 3)

    cv2.namedWindow('contours_img', cv2.WINDOW_NORMAL)
    cv2.imshow("contours_img", temp_img)
    cv2.waitKey(cv_time_wait)

    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)

    # Initialize the detector parameters using default values
    parameters =  cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG


    markerCorners, markerIds, _ = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    if markerIds is not None:
        class_ind+=markerIds.shape[0]

    output_corners, output_id, output_cnter = [], [], 0
    img_make_clone = img.copy()
    # cv2.aruco.drawDetectedMarkers(img_make_clone, markerCorners, markerIds)

    cv2.namedWindow('drawDetectedMarkers', cv2.WINDOW_NORMAL)
    cv2.imshow('drawDetectedMarkers', img_make_clone)
    cv2.waitKey(cv_time_wait)


    return_list_corner_id = []

    for ind in range(len(contours)):
        segregates = []
        if cv2.contourArea(contours[ind]) <=50:
            continue
        internal_mask = np.zeros(mask_real_corners.shape, dtype=np.uint8)


        cv2.drawContours(internal_mask, contours, ind, 1, -1)
        kernel = np.ones((5,5),np.uint8)
        internal_mask = cv2.dilate(internal_mask,kernel,iterations = 1)


        internal_mask = internal_mask* mask_real_corners

        internal_contours, _ = cv2.findContours(
            internal_mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for inner_ind in range(len(internal_contours)):
            internal_internal_mask = np.zeros(
                mask_real_corners.shape, dtype=np.uint8)
            cv2.drawContours(internal_internal_mask,
                             internal_contours, inner_ind, 255, -1)



            internal_internal_mask = internal_internal_mask* mask_real_corners

            cv2.namedWindow("internal_internal_mask", cv2.WINDOW_NORMAL)
            cv2.imshow("internal_internal_mask", internal_internal_mask)
            cv2.waitKey(cv_time_wait)

            #find the center of contours
            M = cv2.moments(internal_contours[inner_ind])
            cX = (M["m10"] / (M["m00"]+1e-18))
            cY = (M["m01"] / (M["m00"]+1e-18))
            segregates.append([cX, cY])


        _segregates  = []

        for x in segregates:
            if(x!=[0,0]):
                _segregates.append((int(x[0]), int(x[1])))
        if len(_segregates) <=2:
            continue


        hull = cv2.convexHull(np.array(_segregates))
        t_segregates = []

        epsilon = 0.01*cv2.arcLength(hull,True)
        hull = cv2.approxPolyDP(hull,epsilon,True)

        for p in hull:
            for x in segregates:
                if( (x[0]-p[0][0])**2 + (x[1] - p[0][1])**2<=10):
                    t_segregates.append(x)

        segregates = t_segregates

        if len(segregates) <4:
            continue

        segregates = order_points(segregates)


        corner_list = []
        for i in segregates:
            corner_list.append((i[0], i[1]))

        if len(corner_list) != 4:
            print("fail", len(corner_list))
            continue


        return_list_corner_id.append([*segregates, 0])



        pad = 80
        tag_size = 224
        h, _ = cv2.findHomography(
            np.array(corner_list), np.array([[pad, pad], [pad, tag_size-pad],[tag_size-pad, tag_size-pad],  [tag_size-pad, 0+pad]]))


        im1Reg = cv2.warpPerspective(img, h, (tag_size, tag_size))


        im2Reg = Image.fromarray(im1Reg)

        ds = dataset_classifier.DirDataset('', '')
        im2Reg = (ds.preprocess(im2Reg))
        with open("bytecode_36h11.pkl", "rb") as f:
            bytecode_36h11 = pickle.load(f)
        out = net((im2Reg.unsqueeze(0).to(device), torch.tensor(bytecode_36h11).to(device)))



        cv2.namedWindow('unrotated_tag', cv2.WINDOW_NORMAL)
        cv2.imshow('unrotated_tag', im1Reg)

        cv2.namedWindow('decoded_tag', cv2.WINDOW_NORMAL)
        cv2.imshow('decoded_tag', (nn.Sigmoid()(out)>0.99).reshape(-1).reshape(10,10).cpu().numpy().astype(np.float32).transpose())
        cv2.waitKey(cv_time_wait)

        rotation = 0

        (h, w) = im1Reg.shape[:2]

        # calculate the center of the image
        center = (w / 2, h / 2)
        scale = 1

        M = cv2.getRotationMatrix2D(center, rotation, scale)
        im3Reg = cv2.warpAffine(im1Reg, M, (h, w))



        inf_ind+=1
        print(inf_ind)

        cv2.putText(im3Reg, "tag_used: "+str(inf_ind), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
        cv2.putText(im3Reg, "class_ind: "+str(class_ind), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        im3Reg[pad,pad] = [0,255,255]
        im3Reg[pad,tag_size-pad] = [0,255,255]
        im3Reg[tag_size-pad,pad] = [0,255,255]
        im3Reg[tag_size-pad,tag_size-pad] = [0,255,255]

        cv2.namedWindow('Unet-prediction', cv2.WINDOW_NORMAL)
        cv2.imshow('Unet-prediction', im3Reg)
        cv2.waitKey(cv_time_wait)

        detected = True




        if markerCorners is not None and markerIds is not None:
            for corners, id in zip(markerCorners, markerIds):
                corner = corners[0]


                corner_out = corner

                real_corner_list = []
                fake_corner_list = []
                for corner in corner:
                    min_dist = 10.0
                    closest = []
                    for inner_corner in corner_list:
                        dist = (inner_corner[0] - corner[0])**2 + (inner_corner[1] - corner[1])**2
                        dist = math.sqrt(dist);
                        if(dist<min_dist):
                            closest = inner_corner
                            min_dist = dist;

                    if(closest != []):
                        real_corner_list.append(closest);
                        fake_corner_list.append(corner);

                if len(real_corner_list) == 4:
                    output_cnter+=1
                    output_corners.append(real_corner_list)
                    # output_corners.append(fake_corner_list)
                    output_id.append(id)






        print(corner_list)


    global_output_cnter.append(output_cnter)
    global_output_corners.append(output_corners)
    global_output_id.append(output_id)

    print(markerCorners)


    # ret = markerCorners[0] if len(markerCorners)>0 else []
    # return ret, len(markerCorners)>0
    return return_list_corner_id, detected


def predict(net, img, device='cuda', threshold=0.5, kernel =1024, stride =1024):
    with torch.no_grad():
        ds = DirDataset('', '')
        _img = (ds.preprocess(img))


        _img = _img.unsqueeze(0)
        _img = _img.to(device=device, dtype=torch.float32)



        return (net(_img)*1024).reshape((4,2)).cpu().numpy()


def return_corner_coords_with_id(img,net_unet, net_bytecode, args):
    """
        Return a list of array with 5 element in each [corner_1, corner_2, corner_3, corner_4, id]
    """


    return predict(net_unet, img,threshold=args.segmentation_threshold,kernel=args.kernel, stride = args.stride,  device=args.device)








def main(args):
    with torch.no_grad():


        net_unet = Unet.load_from_checkpoint(args.checkpoint, strict = False)
        net_unet.to(args.device)
        net_unet.eval()


        # input_sample = torch.randn((1,3, 1024, 1024))
        # net_unet.to_onnx("model_unet.onnx", input_sample, export_params=True)

        net_bytecode = Resnet.load_from_checkpoint(args.checkpoint_bit)
        net_bytecode.freeze()
        net_bytecode.to(args.device)
        net_bytecode.eval()

        img_list = sorted([str(item) for item in glob.glob(args.img + "*.jpg")])[:1000]
        pkl_list = sorted([str(item) for item in glob.glob(args.img + "*.pkl")])[:1000]

        for img_str, pkl_str in zip(img_list, pkl_list):
            im_size = 1024
            img = Image.open(img_str).convert('RGB')
            ind = img_str[-7:-4]



            img = img.resize((im_size, im_size))

            open_cv_image = np.array(img)
            # Convert RGB to BGR
            open_cv_image = open_cv_image[:, :, ::-1].copy()

            # size = 15

            # # generating the kernel
            # kernel_motion_blur = np.zeros((size, size))
            # kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
            # kernel_motion_blur = kernel_motion_blur / size

            # # applying the kernel to the input image
            # open_cv_image = cv2.filter2D(open_cv_image, -1, kernel_motion_blur)


            img = Image.fromarray(open_cv_image)

            corners_with_id = return_corner_coords_with_id(img, net_unet, net_bytecode, args)
            corners = corners_with_id[:4]

            corners_pkl = []
            with open(pkl_str, "rb") as f:
                corners_pkl = np.array(pickle.load(f)).reshape((-1,2)).tolist()
            print(corners)
            print(corners_pkl)
            dst_sm = 0.00

            for c in corners[0][:4]:
                mn = (1000000000000, (0,0))
                for ic in corners_pkl:
                    print(c, ic)
                    diff = 0.0
                    mn = min(mn, ((ic[0]/2 - c[0] -diff  )* (ic[0]/2 - c[0] -diff ) + (ic[1]/2 - c[1] -diff )* (ic[1]/2 - c[1]  -diff), (ic[0]/2 -c[0] , ic[1]/2 - c[1]) ))
                    mn = min(mn, ((ic[0] - c[0] -diff  )* (ic[0] - c[0] -diff  ) + (ic[1] - c[1] -diff )* (ic[1] - c[1]-diff ), (ic[0]/2 -c[0] , ic[1]/2 - c[1]) ))

                dst_sm += math.sqrt(mn[0])
                # with open("file_corners_unet_diff.csv", "a") as f:
                #      f.write(str(ind)+","+ str(mn[1][0]) + ","+str(mn[1][1]) + " \n")

            dst_sm /= 4
            print(dst_sm)
            # with open("file_corners_unet.csv", "a") as f:
            #      f.write(str(ind)+","+ str(dst_sm) + " \n")


    with open('outputs/unet_corner_direct.pkl', "wb") as f:
        pickle.dump((global_output_corners, global_output_id, global_output_cnter, (1024,1024)), f)










if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help = False)
    parent_parser.add_argument('--checkpoint', required=True, help = "Network for segmentation")
    parent_parser.add_argument('--checkpoint_bit', required=True, help = "Network for segmentation")
    parent_parser.add_argument('--device', default = "cuda", type = str, required=False, help = "Use GPU or not : cuda or cpu" )
    parent_parser.add_argument('--kernel', default = 1024, type = int , required=False, help = "Image size for one patch to Unet" )
    parent_parser.add_argument('--stride', default =1024, type=int, required=False, help = "Image stride for one patch to Unet" )
    parent_parser.add_argument('--segmentation_threshold', default = 0.1, type=float, required=False, help = "Threshold for using in segmentation" )
    parent_parser.add_argument('--corner_threshold', default = 0.1, type = float, required=False, help = "Threshold for using in corner " )
    parent_parser.add_argument('--img', required=True, type = str)

    parser = Unet.add_model_specific_args(parent_parser)
    args = parser.parse_args()

    main(args)