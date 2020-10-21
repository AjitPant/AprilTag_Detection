import os
from argparse import ArgumentParser
import glob
import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
import dataset_classifier
import dataset_classifier_id

from Unet import Unet
from classifier import Resnet
from classifier_id import Resnet as Resnet_id
from corners_to_crop import crop_to_corners
from dataset import DirDataset

from torchvision import datasets, models, transforms


from scipy.spatial import distance as dist
from scipy import stats
import pickle


import torch



import operator
import math
from functools import reduce


cv_time_wait = 1
def nonzero_mode(arr):
    return stats.mode(arr[np.nonzero(arr)]).mode

from scipy.spatial import distance as dist
import numpy as np
import cv2

inf_ind = 0
def order_points(pts):
	coords = pts
	center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
	out = sorted(coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
	return np.array(out, dtype="float32")


def reduce_to_tags(net, net_id, img, response_1, response_2,homography_mat, filename, hparams, coords_collection):
    global inf_ind
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    mask_segmentation = response_1
    mask_corners = response_2
    segregates = []
    print(mask_corners.max())
    cv2.waitKey(cv_time_wait)
    mask_corners =  (mask_corners>0.5).astype(np.uint8)
    print(mask_corners.shape)


    cv2.namedWindow('mask_segmentation', cv2.WINDOW_NORMAL)
    cv2.imshow("mask_segmentation", mask_segmentation*255)
    cv2.waitKey(cv_time_wait)

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
    print(mask_real_corners)
    print(mask_real_corners.shape)
    # mask_real_corners = 4- np.argmax(mask_corners, axis = 0).astype(np.uint8)
    # print(mask_real_corners)

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
    for ind in range(len(contours)):
        segregates = []
        internal_mask = np.zeros(mask_real_corners.shape, dtype=np.uint8)

        cv2.drawContours(internal_mask, contours, ind, 255, -1)
        # cv2.drawContours(internal_mask, contours, ind, 255, 10)

        cv2.namedWindow('internal_mask', cv2.WINDOW_NORMAL)
        cv2.imshow("internal_mask", internal_mask)
        cv2.waitKey(cv_time_wait)
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
            cX = (M["m10"] / (M["m00"]+1e-18))
            cY = (M["m01"] / (M["m00"]+1e-18))
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


        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # find Harris corners
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray,2,3,0.04)
        # dst = cv2.dilate(dst,None)
        ret, dst = cv2.threshold(dst,0.001*dst.max(),255,0)
        dst = np.uint8(dst)

        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.0001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

        # Now draw them
        res = np.hstack((centroids,corners))
        print(corners)


        real_corner_list = []
        for corner in corner_list:
            min_dist = 10000000.0
            closest = []
            for inner_corner in corners:
                dist = (inner_corner[0] - corner[0])**2 + (inner_corner[1] - corner[1])**2
                dist = math.sqrt(dist);
                if(dist<min_dist):
                    closest = inner_corner
                    min_dist = dist;
            if(min_dist < 0):
                real_corner_list.append(closest);
            else:
                real_corner_list.append(corner);
        # corner_list = real_corner_list



        res = np.int0(res)
        # img[res[:,1],res[:,0]]=[0,250,255]
        # img[res[:,3],res[:,2]] = [0,255,0]
        img_clone = img.copy()
        for corner in corner_list:
            img_clone[np.int(corner[1]),np.int(corner[0])] =[0,0,255]

        cv2.namedWindow('subpixel5.png', cv2.WINDOW_NORMAL)
        cv2.imshow('subpixel5.png',img_clone)
        cv2.waitKey(cv_time_wait)

        # print(corner_list)
        pad = 30
        tag_size = 224
        h, status = cv2.findHomography(
            np.array(corner_list), np.array([[pad, pad], [pad, tag_size-pad],[tag_size-pad, tag_size-pad],  [tag_size-pad, 0+pad]]))
        # corner_list = np.array([[pad, pad], [pad, tag_size-pad],[tag_size-pad, tag_size-pad],  [tag_size-pad, 0+pad]])

        height, width, channels = img.shape

        im1Reg = cv2.warpPerspective(img, h, (tag_size, tag_size))

        cv2.namedWindow('unrotated_tag', cv2.WINDOW_NORMAL)
        cv2.imshow('unrotated_tag', im1Reg)
        cv2.waitKey(cv_time_wait)

        # im1Reg = cv2.cvtColor(im1Reg, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(im1Reg,(5,5),0)
        # reg,im1Reg = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # im1Reg = cv2.filter2D(im1Reg, -1, kernel)

        # cv2.namedWindow('threshold_tag', cv2.WINDOW_NORMAL)
        # cv2.imshow('threshold_tag', im1Reg)
        # cv2.waitKey(cv_time_wait)

        im2Reg = Image.fromarray(im1Reg)

        ds = dataset_classifier.DirDataset('', '')
        im2Reg = (ds.preprocess(im2Reg))
        out = net(im2Reg.unsqueeze(0).to(device))
        print(out)
        rotation = ((np.argmax(out.squeeze(0).cpu())+2)*-90).item()
        print(rotation)

        (h, w) = im1Reg.shape[:2]

        # calculate the center of the image
        center = (w / 2, h / 2)
        scale = 1

        M = cv2.getRotationMatrix2D(center, rotation, scale)
        im3Reg = cv2.warpAffine(im1Reg, M, (h, w))

        # cv2.namedWindow('rotated_tag', cv2.WINDOW_NORMAL)
        # cv2.imshow('rotated_tag', im3Reg)
        # cv2.waitKey(cv_time_wait)

        im31Reg = Image.fromarray(im1Reg)

        ds = dataset_classifier_id.DirDataset('', '')
        im31Reg = (ds.preprocess(im31Reg))
        out = net_id(im31Reg.unsqueeze(0).to(device))

        glb_id = ((np.argmax(out.squeeze(0).cpu()))).item()
        if(torch.softmax(out,dim = 1)[0][glb_id]<0.5):
            continue;

        if( glb_id == 587):
            continue
        inf_ind+=1

        cv2.putText(im3Reg, "id: "+str(glb_id), (tag_size//2-20, tag_size//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100), 2)
        cv2.putText(im3Reg, "tag_used: "+str(inf_ind), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        im3Reg[pad,pad] = [0,255,255]
        im3Reg[pad,tag_size-pad] = [0,255,255]
        im3Reg[tag_size-pad,pad] = [0,255,255]
        im3Reg[tag_size-pad,tag_size-pad] = [0,255,255]

        cv2.namedWindow('Unet-prediction', cv2.WINDOW_NORMAL)
        cv2.imshow('Unet-prediction', im3Reg)
        cv2.waitKey(cv_time_wait)



        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)

        # Initialize the detector parameters using default values
        parameters =  cv2.aruco.DetectorParameters_create()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        # parameters.maxErroneousBitsInBorderRate = 0.8
        # parameters.errorCorrectionRate = 0.8
        # parameters.aprilTagMaxLineFitMse = 100
        # parameters.perspectiveRemoveIgnoredMarginPerCell = 0.43
        # parameters.perspectiveRemovePixelPerCell = 10


        # markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
        # # markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(im1Reg, dictionary, parameters=parameters)
        # # print(im1Reg)
        # print(im1Reg.max())
        # print(im1Reg.min())

        # print(markerCorners)
        # print(rejectedCandidates)
        # pad = pad
        # if markerCorners is not None and markerIds is not None:
        #     for corners, id in zip(markerCorners, markerIds):
        #         print(corners)
        #         corner = corners[0]
        #         print(corner)
        #         h, status = cv2.findHomography(
        #                     np.array(corner), np.array([[pad, pad], [tag_size-pad, pad], [tag_size-pad, tag_size-pad], [pad, tag_size-pad]]))
        #         height, width, channels = img.shape
        #         im1Regg = cv2.warpPerspective(img, h, (tag_size, tag_size))
        #         # im1Regg = cv2.warpPerspective(im1Reg, h, (tag_size, tag_size))

        #         im1Regg[pad,pad] = [0,255,255]
        #         im1Regg[pad,tag_size-pad] = [0,255,255]
        #         im1Regg[tag_size-pad,pad] = [0,255,255]
        #         im1Regg[tag_size-pad,tag_size-pad] = [0,255,255]


        #         for corners in corner:
        #             img_clone[np.int(corners[1]),np.int(corners[0])] =[255,0,255]
        #         cv2.namedWindow('subpixel_classical.png', cv2.WINDOW_NORMAL)
        #         cv2.imshow('subpixel_classical.png',img_clone)
        #         cv2.waitKey(cv_time_wait)

        #         if(id==glb_id):

        #             cv2.putText(im1Regg, "id: "+str(id), (tag_size//2-20, tag_size//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        #             cv2.putText(im1Regg, "tag_used: "+str(inf_ind), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

        #             cv2.namedWindow('classical_detection', cv2.WINDOW_NORMAL)
        #             cv2.imshow('classical_detection', im1Regg)
        #             cv2.waitKey(cv_time_wait)


        #         corner_out = corner

        #         print(corner_list)
        #         print(corner_out)
        real_corner_list = []
        fake_corner_list = []
        print(coords_collection);
        for mike_corner in coords_collection:
            for corner in mike_corner[0]:
                corner = corner*img.shape[0]
                min_dist = 10.0
                closest = []
                for inner_corner in corner_list:
                    dist = (inner_corner[0] - corner[0])**2 + (inner_corner[1] - corner[1])**2
                    dist = math.sqrt(dist);
                    if(dist<min_dist):
                        closest = inner_corner
                        min_dist = dist;

                print(min_dist)
                print(corner)
                print(corner_list)
                if(closest != []):
                    real_corner_list.append(closest);
                    fake_corner_list.append(corner);


            if len(real_corner_list) == 4:
                # print(real_corner_list, corner_out)
                for p1,p2 in zip(real_corner_list, fake_corner_list  ):
                    with open("./data_out/data.txt", "a") as diff_file:

                        diff_file.write(str(p1[0])+" "+str(p1[1]) + "," + str(p2[0]) + " " + str(p2[1])+"\n")
                    x_diff = p1[0] - p2[0]
                    y_diff = p1[1] - p2[1]
                    with open("./data_out/diff_file_" + str(id) + ".csv", "a") as diff_file:

                        diff_file.write(str(x_diff) + "," + str(y_diff)+"\n")


        # for corners in rejectedCandidates:
        #     corner = corners[0]
        #     print(corners)
        #     h, status = cv2.findHomography(
        #                 np.array(corner), np.array([[pad, pad], [pad, tag_size-pad], [tag_size-pad, tag_size-pad], [tag_size-pad, 0+pad]]))
        #     height, width, channels = img.shape
        #     im1Regg = cv2.warpPerspective(im1Reg, h, (tag_size, tag_size))
        #     cv2.namedWindow('classical_detection', cv2.WINDOW_NORMAL)
        #     cv2.imshow('classical_detection', im1Regg)
        #     cv2.waitKey(cv_time_wait)




        # print(corner_list)

        # cv2.imwrite(os.path.join(args.out_folder, 'simg',
        #                          filename[:-4] + "_" + str(index) + '.jpg'), im1Reg)


        # print(f"{mask_corners[int(corner_list[0][1])][int(corner_list[0][0])]}")
        # with open(os.path.join(args.out_folder, 'simg',filename[:-4] + "_" + str(index) + '.txt'), "w") as text_file:
        #     print(f"{mask_corners[int(corner_list[0][1])][int(corner_list[0][0])]}", file=text_file)

        index = index + 1



def predict(net, img, device='cpu', threshold=0.5):
    ds = DirDataset('', '')
    _img = (ds.preprocess(img))


    # cv2.imshow("predict", _img.cpu().numpy().transpose((2, 1, 0)))
    # cv2.waitKey(cv_time_wait)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(hparams.id_net)
    net_2 = Resnet.load_from_checkpoint(hparams.id_net)
    net_2.freeze()
    net_2.to(device)
    net_2.eval()

    net_id = Resnet_id.load_from_checkpoint(hparams.id_classifier_net)
    net_id.freeze()
    net_id.to(device)
    net_id.eval()
    img_list = [str(item) for item in glob.glob(hparams.img)]

    for img_str in img_list:
        print(img_str)
        pkl_str = "."+img_str.split('.')[1]  + str('.pkl')
        with open(pkl_str, 'rb') as f:
            coords_collection = pickle.load(f)

        im_size = 768
        img = Image.open(img_str).convert('RGB')

        width, height = img.size
        # width, height = (im_size, im_size)


        homography_mat, status = cv2.findHomography(
            np.array([[0, 0], [0, im_size], [im_size, im_size], [im_size, 0]]),
            np.array([[0, 0], [0, width], [height, width], [height, 0]]))
        img = img.resize((im_size, im_size))
        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        img = Image.fromarray(open_cv_image)

        mask, _mask = predict(net, img, device=device)


        # img = Image.open(img_str).convert('RGB')
        # open_cv_image = np.array(img)
        # # Convert RGB to BGR
        # open_cv_image = open_cv_image[:, :, ::-1].copy()
        # img = Image.fromarray(open_cv_image)

        # # destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)


        # # mask = 4 - mask.argmax(axis = 0);
        # print(open_cv_image.shape[:2])
        # print(mask.shape)
        # print(_mask.shape)
        # print((open_cv_image.shape[0], open_cv_image.shape[1]))
        # print(mask.dtype);
        # print(_mask.dtype);
        # _mask = _mask.astype(np.float32)

        # mask = np.squeeze(mask, axis = 0)
        # # _mask = _mask.squeeze(0)
        # mask = cv2.resize(mask, dsize=(open_cv_image.shape[1], open_cv_image.shape[0]), interpolation = cv2.INTER_AREA)
        # _mask = cv2.resize(_mask, dsize=(open_cv_image.shape[1], open_cv_image.shape[0]), interpolation = cv2.INTER_AREA)
        # mask = np.expand_dims(mask, axis = 0)
        # # _mask = _mask.unsqueeze(0)


        img = np.array(img)

        # print(img.dtype)

        # mask = mask.astype(np.uint8)
        _mask = _mask.astype(np.uint8)
        # crop_to_corners(identification_net, img, [_mask, mask], device)
        reduce_to_tags(net_2, net_id,img, _mask, mask,homography_mat, '.', hparams, coords_collection)



if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help = False)
    parent_parser.add_argument('--checkpoint', required=True, help = "Network for segmentation")
    parent_parser.add_argument('--id_net', required=True, help = "Network for corner classification")
    parent_parser.add_argument('--id_classifier_net', required=True, help = "Network for tag classification")
    # parent_parser.add_argument('--fit_out_net', required=True, help = "Network for tag classification")
    parent_parser.add_argument('--img', required=True, type = str)

    parser = Unet.add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)
