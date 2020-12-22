import os
import pickle
from torch import nn
from argparse import ArgumentParser
import glob
import cv2
import numpy as np
import torch

from PIL import Image
import dataset_classifier

from Unet import Unet
from classifier import Resnet
from dataset import DirDataset

from torchvision import datasets, models, transforms


from scipy import stats





import operator
import math
from functools import reduce


cv_time_wait = 0

from scipy.spatial import distance as dist

inf_ind = 0
class_ind = 0
def order_points(pts):
	coords = pts
	center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
	out = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
	return np.array(out, dtype="float32")


global_output_corners, global_output_id, global_output_cnter = [], [], []
def reduce_to_tags(net, net_id, img, response_1, response_2, filename, hparams):
    global inf_ind
    global class_ind
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    filename = filename.split('/')[-1]

    mask_segmentation = response_1
    mask_corners = response_2
    segregates = []

    mask_corners =  (mask_corners>0.005).astype(np.uint8)

    kernel = np.ones((5,5),np.uint8)
    mask_segmentation = cv2.erode(mask_segmentation,kernel,iterations = 2)


    cv2.namedWindow('mask_segmentation', cv2.WINDOW_NORMAL)
    cv2.imshow("mask_segmentation", mask_segmentation*255)
    cv2.waitKey(cv_time_wait)

    mask_real_corners = np.zeros(mask_corners.shape[1:], dtype=np.uint8)

    mask_real_corners = (mask_corners!=0).astype(np.uint8)

    cv2.namedWindow('mask_garbage', cv2.WINDOW_NORMAL)
    cv2.imshow("mask_garbage", mask_real_corners.astype(np.float32)*60)
    cv2.waitKey(cv_time_wait)

    contours, _ = cv2.findContours(
        mask_segmentation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    temp_img = img.copy()
    coords = np.argwhere(mask_corners > 0)

    cv2.drawContours(temp_img, contours, -1, (0, 255, 0), 3)

    cv2.namedWindow('contours_img', cv2.WINDOW_NORMAL)
    cv2.imshow("contours_img", temp_img)
    cv2.waitKey(cv_time_wait)
    index = 0

    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)

    # Initialize the detector parameters using default values
    parameters =  cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG


    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

    if markerIds is not None:
        class_ind+=markerIds.shape[0]


    img_make_clone = img.copy()
    cv2.aruco.drawDetectedMarkers(img_make_clone, markerCorners, markerIds);

    cv2.namedWindow('drawDetectedMarkers', cv2.WINDOW_NORMAL)
    cv2.imshow('drawDetectedMarkers', img_make_clone)
    cv2.waitKey(cv_time_wait)

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # # find Harris corners
    # gray = np.float32(gray)
    # dst = cv2.cornerHarris(gray,2,3,0.04)
    # # dst = cv2.dilate(dst,None)
    # ret, dst = cv2.threshold(dst,0.001*dst.max(),255,0)
    # dst = np.uint8(dst)

    # # find centroids
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # # define the criteria to stop and refine the corners
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)
    # corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # # Now draw them
    # res = np.hstack((centroids,corners))
    # print(corners)


    output_corners, output_id, output_cnter = [], [], 0

    for ind in range(len(contours)):
        segregates = []
        if cv2.contourArea(contours[ind]) <=50:
            continue
        internal_mask = np.zeros(mask_real_corners.shape, dtype=np.uint8)


        cv2.drawContours(internal_mask, contours, ind, 255, -1)
        kernel = np.ones((5,5),np.uint8)
        internal_mask = cv2.dilate(internal_mask,kernel,iterations = 4)

        # cv2.namedWindow('internal_mask', cv2.WINDOW_NORMAL)
        # cv2.imshow("internal_mask", internal_mask)
        # cv2.waitKey(cv_time_wait)

        internal_mask = cv2.bitwise_and(
            internal_mask, mask_real_corners.astype(np.uint8))

        internal_contours, _ = cv2.findContours(
            internal_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for inner_ind in range(len(internal_contours)):
            internal_internal_mask = np.zeros(
                mask_real_corners.shape, dtype=np.uint8)
            cv2.drawContours(internal_internal_mask,
                             internal_contours, inner_ind, 255, -1)
            internal_internal_mask = cv2.bitwise_and(
                internal_internal_mask, mask_real_corners.astype(np.uint8))

            # cv2.namedWindow('internal_internal_mask', cv2.WINDOW_NORMAL)
            # cv2.imshow("internal_internal_mask", internal_internal_mask)
            # cv2.waitKey(cv_time_wait)

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

        epsilon = 0.05*cv2.arcLength(hull,True)
        hull = cv2.approxPolyDP(hull,epsilon,True)
        print(segregates)

        for p in hull:
            for x in segregates:
                if( (x[0]-p[0][0])**2 + (x[1] - p[0][1])**2<=10):
                    t_segregates.append(x);

        segregates = t_segregates

        # if len(segregates) < 4:
        #     continue


        segregates = order_points(segregates)
        # print(len(segregates))
        # if len(segregates) != 4:
        #     continue

        corner_list = []
        # print(segregates)
        for i in segregates:
            corner_list.append((i[0], i[1]))

        if len(corner_list) != 4:
            # cv2.waitKey(0)
            continue



        # real_corner_list = []
        # for corner in corner_list:
        #     min_dist = 10000000.0
        #     closest = []
        #     for inner_corner in corners:
        #         dist = (inner_corner[0] - corner[0])**2 + (inner_corner[1] - corner[1])**2
        #         dist = math.sqrt(dist);
        #         if(dist<min_dist):
        #             closest = inner_corner
        #             min_dist = dist;
        #     if(min_dist < 0):
        #         real_corner_list.append(closest);
        #     else:
        #         real_corner_list.append(corner);
        # subpix_corner_list = real_corner_list



        # res = np.int0(res)
        # # img[res[:,1],res[:,0]]=[0,250,255]
        # # img[res[:,3],res[:,2]] = [0,255,0]
        # img_clone = img.copy()
        # for corner in corner_list:
        #     img_clone[np.int(corner[1]),np.int(corner[0])] =[0,0,255]

        # cv2.namedWindow('subpixel5.png', cv2.WINDOW_NORMAL)
        # cv2.imshow('subpixel5.png',img_clone)
        # cv2.waitKey(cv_time_wait)

        # print(corner_list)


        pad = 40
        tag_size = 224
        h, status = cv2.findHomography(
            np.array(corner_list), np.array([[pad, pad], [pad, tag_size-pad],[tag_size-pad, tag_size-pad],  [tag_size-pad, 0+pad]]))
        # corner_list = np.array([[pad, pad], [pad, tag_size-pad],[tag_size-pad, tag_size-pad],  [tag_size-pad, 0+pad]])

        height, width, channels = img.shape

        im1Reg = cv2.warpPerspective(img, h, (tag_size, tag_size))


        # # im1Reg = cv2.cvtColor(im1Reg, cv2.COLOR_BGR2GRAY)
        # # blur = cv2.GaussianBlur(im1Reg,(5,5),0)
        # # reg,im1Reg = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # # im1Reg = cv2.filter2D(im1Reg, -1, kernel)

        # # cv2.namedWindow('threshold_tag', cv2.WINDOW_NORMAL)
        # # cv2.imshow('threshold_tag', im1Reg)
        # # cv2.waitKey(cv_time_wait)

        im2Reg = Image.fromarray(im1Reg)

        ds = dataset_classifier.DirDataset('', '')
        im2Reg = (ds.preprocess(im2Reg))



        out = net(im2Reg.unsqueeze(0).to(device))


        print(torch.round(nn.Sigmoid()(out)).reshape(10,10))

        cv2.namedWindow('unrotated_tag', cv2.WINDOW_NORMAL)
        cv2.imshow('unrotated_tag', im1Reg)
        cv2.waitKey(cv_time_wait)

        rotation = 0

        (h, w) = im1Reg.shape[:2]

        # calculate the center of the image
        center = (w / 2, h / 2)
        scale = 1

        M = cv2.getRotationMatrix2D(center, rotation, scale)
        im3Reg = cv2.warpAffine(im1Reg, M, (h, w))

        # # cv2.namedWindow('rotated_tag', cv2.WINDOW_NORMAL)
        # # cv2.imshow('rotated_tag', im3Reg)
        # # cv2.waitKey(cv_time_wait)

        im31Reg = Image.fromarray(im1Reg)

        inf_ind+=1
        print(inf_ind)
        with open("./data_out/file_n.txt", "a") as diff_file:

            diff_file.write(str(inf_ind)+"\n")

        # cv2.putText(im3Reg, "id: "+str(glb_id), (tag_size//2-20, tag_size//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100), 2)
        cv2.putText(im3Reg, "tag_used: "+str(inf_ind), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
        cv2.putText(im3Reg, "class_ind: "+str(class_ind), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        im3Reg[pad,pad] = [0,255,255]
        im3Reg[pad,tag_size-pad] = [0,255,255]
        im3Reg[tag_size-pad,pad] = [0,255,255]
        im3Reg[tag_size-pad,tag_size-pad] = [0,255,255]

        cv2.namedWindow('Unet-prediction', cv2.WINDOW_NORMAL)
        cv2.imshow('Unet-prediction', im3Reg)
        cv2.waitKey(cv_time_wait)






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


        #         if len(real_corner_list) == 4:
        #             print(real_corner_list, corner_out)
        #             for p1,p2 in zip(real_corner_list, fake_corner_list  ):
        #                 with open("./data_out/data.txt", "a") as diff_file:

        #                     diff_file.write(str(p1[0])+","+str(p1[1]) + "," + str(p2[0]) + "," + str(p2[1]) + ","+ str(id)[1:-1]+"," + str(p2[0]-p1[0])+"," + str(p2[1]-p1[1]) +"\n")
        #                 x_diff = p1[0] - p2[0]
        #                 y_diff = p1[1] - p2[1]
        #                 with open("./data_out/diff_file_" + str(id) + ".csv", "a") as diff_file:

        #                     diff_file.write(str(p1[0])+","+str(p1[1]) + "," + str(p2[0]) + "," + str(p2[1]) + ","+ str(id)[1:-1]+"," + str(p2[0]-p1[0])+"," + str(p2[1]-p1[1]) +"\n")

        #                 with open("./data_out/img/" + filename + ".csv", "a") as diff_file:

        #                     diff_file.write(str(p1[0])+","+str(p1[1]) + "," + str(p2[0]) + "," + str(p2[1]) + ","+ str(id)[1:-1]+"," + str(p2[0]-p1[0])+"," + str(p2[1]-p1[1]) +"\n")






        # print(corner_list)

        # cv2.imwrite(os.path.join(args.out_folder, 'simg',
        #                          filename[:-4] + "_" + str(index) + '.jpg'), im1Reg)


        # print(f"{mask_corners[int(corner_list[0][1])][int(corner_list[0][0])]}")
        # with open(os.path.join(args.out_folder, 'simg',filename[:-4] + "_" + str(index) + '.txt'), "w") as text_file:
        #     print(f"{mask_corners[int(corner_list[0][1])][int(corner_list[0][0])]}", file=text_file)

        index = index + 1

    global_output_cnter.append(output_cnter)
    global_output_corners.append(output_corners)
    global_output_id.append(output_id)



def predict(net, img, device='cuda', threshold=0.5, kernel =1024, stride =768):
    with torch.no_grad():
        ds = DirDataset('', '')
        _img = (ds.preprocess(img))


        _img = _img.unsqueeze(0)
        _img = _img.to(device=device, dtype=torch.float32)

        mask =  torch.zeros(_img.shape[2:], device = device, dtype = torch.float32)
        _mask =  torch.zeros(_img.shape[2:], device = device, dtype = torch.float32)
        rows,cols = _img.shape[2:]

        with torch.no_grad():

            for start_row in range(0,rows, stride):
                for start_col in range(0,cols,stride):

                    patch = _img[:,:, start_row:start_row+kernel, start_col:start_col+kernel]

                    patch_height, patch_width = patch.shape[2:]
                    m = nn.ZeroPad2d((0,  max(0, kernel - patch.shape[3]) , 0, max(0, kernel -patch.shape[2])))
                    print(patch.shape)
                    patch = m(patch)
                    print(patch.shape)
                    o = net(patch)

                    _o = o[:, 1, :, :]
                    o = o[:, :1,:,:]

                    probs = torch.sigmoid(o)
                    probs = probs.squeeze(0)
                    mask_patch = probs

                    _probs = torch.sigmoid(_o)
                    _probs = _probs.squeeze(0)
                    _mask_patch = _probs
                    print(mask_patch.shape)
                    print(_mask_patch.shape)
                    print(mask_patch[:patch_height, :patch_width].shape)
                    print(_mask_patch[:patch_height, :patch_width].shape)

                    mask[start_row:start_row+kernel, start_col:start_col+kernel] = torch.max(mask[start_row:start_row+kernel, start_col:start_col+kernel], mask_patch[:, :patch_height, :patch_width]);
                    _mask[start_row:start_row+kernel, start_col:start_col+kernel] = torch.max(_mask[start_row:start_row+kernel, start_col:start_col+kernel], _mask_patch[:patch_height, :patch_width]);
        return (mask.cpu().numpy(), _mask.cpu().numpy() > threshold )


def mask_to_image(mask):
    return Image.fromarray(( mask ).astype(np.uint8))


def main(hparams):
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print("outmark")
        net = Unet.load_from_checkpoint(hparams.checkpoint)
        print("outmark_done")
        # net.freeze()
        net.to(device)
        net.eval()



        print(hparams.checkpoint_bit)
        net_2 = Resnet.load_from_checkpoint(hparams.checkpoint_bit)
        net_2.freeze()
        net_2.to(device)
        net_2.eval()

        img_list = [str(item) for item in glob.glob(hparams.img)]

        for img_str in img_list:
            im_size = 1024
            img = Image.open(img_str).convert('RGB')



            img = img.resize((im_size, im_size))

            width, height = img.size
            open_cv_image = np.array(img)
            # Convert RGB to BGR
            # open_cv_image = open_cv_image[:, :, ::-1].copy()

            img = Image.fromarray(open_cv_image)

            mask, _mask = predict(net, img, device=device)

            img = np.array(img)

            _mask = _mask.astype(np.uint8)

            reduce_to_tags(net_2, net_2,img, _mask, mask, img_str, hparams)



    with open('outputs/unet_corner_direct.pkl', "wb") as f:
        pickle.dump((global_output_corners, global_output_id, global_output_cnter, img.shape), f)
if __name__ == '__main__':
    print("hi")
    parent_parser = ArgumentParser(add_help = False)
    parent_parser.add_argument('--checkpoint', required=True, help = "Network for segmentation")
    parent_parser.add_argument('--checkpoint_bit', required=True, help = "Network for segmentation")
    parent_parser.add_argument('--id_net', required=True, help = "Network for corner classification")
    # parent_parser.add_argument('--fit_out_net', required=True, help = "Network for tag classification")
    parent_parser.add_argument('--img', required=True, type = str)

    print("hi")
    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()
    print("hi")

    main(hparams)
