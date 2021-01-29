import os, glob, cv2
from math import exp
import pickle

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import copy
from torchvision import transforms
import albumentations as A

class DirDataset(Dataset):

    def __init__(self, img_dir, mask_dir, scale=1):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.scale = scale
        original_height = 512*2
        original_width = 512*2
        print(self.img_dir)

        self.aug = A.Compose([

            A.OneOf([







               A.RandomSizedCrop(min_max_height=(original_height//2, original_height),
                                  height=original_height//1, width=original_width//1, p=1.0),

            ], p=0.0),

            A.PadIfNeeded(min_height=original_height, min_width=original_width, p = 1.0, border_mode=cv2.BORDER_CONSTANT, value = 0),
            A.OneOf([
                A.Blur((5,25), p = 0.5),
                A.MotionBlur((5,25),p =  0.5),

            ], p=0.0),

            A.OneOf([
                A.ToGray(),
                A.ChannelDropout(),
            ], p=0.5),


            A.OneOf([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
            ], p=0.0),

            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=180, p=0.5),
            ], p=0.0),
            A.CLAHE(p = 0.2),
            A.GaussNoise(p = 0.5),
            A.OneOf([
                A.RandomRain(),
                A.RandomFog( fog_coef_lower = 0.1, fog_coef_upper = 0.3),
            ], p=0.3),
            A.OneOf([
                A.RandomShadow(p=0.4, num_shadows_upper=5),
                A.RandomSunFlare(src_radius=20,p = 0.2 ),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RGBShift(p=0.5),
                A.RandomGamma(p=0.8)
            ], p=0.4),
        ], p=0.9,
            additional_targets={
            'image': 'image',

            'mask0': 'mask',
            'mask1': 'mask',
            'keypoints': 'keypoints',
        }, keypoint_params=A.KeypointParams(format='xy'))

        try:
            self.ids = (sorted([os.path.splitext(s)[0]
                                for s in os.listdir(self.img_dir) if os.path.splitext(s)[1] == '.jpg']))

        except FileNotFoundError:
            self.ids = []
        print("extracted_ ids cnt : "+str(len(self.ids)))
    def __len__(self):
        return len(self.ids)

    def preprocess(self, img, mask=False):

        if not mask:
            trans = transforms.Compose([
                transforms.ToTensor(),

                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ])

            img = trans(img).float()

        return img

    def __getitem__(self, i):
        idx = self.ids[i]
        img_files = [os.path.join(self.img_dir, idx + '.jpg')]
        mask_files = [os.path.join(self.mask_dir, idx + '_0.png'),
                      os.path.join(self.mask_dir, idx + '_5.png'), ]
        keypoints_file = os.path.join(self.img_dir, idx + '.pkl')

        assert all([os.path.exists(path) for path in img_files]), 'image files missing'
        assert all([os.path.exists(path) for path in mask_files]), 'mask files missing'
        assert os.path.exists(keypoints_file), 'keypoints files missing'

        img = cv2.imread(img_files[0])
        mask = [cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                for mask_file in mask_files]

        # img = cv2.resize(img, (512,512))
        # mask[0] = cv2.resize(mask[0], (512,512))
        # mask[1] = cv2.resize(mask[1], (512,512))

        with open(keypoints_file, "rb") as f:
            keypoints = (np.array(pickle.load(f)).reshape((-1,2))).tolist()

        augmented = self.aug(image=img, mask0=mask[0], mask1=mask[1], keypoints = keypoints)
        img = augmented['image']

        mask[1] = augmented['mask1']
        mask[0] = augmented['mask0'].astype(np.float32)

        keypoints = augmented['keypoints']




        mask[0].fill(0)

        d = 16

        for point in keypoints:
            # point = [point[0]/2, point[1]/2]
            for x in range(max(0, int(point[1])-d),min(img.shape[0], int(point[1])+d+1)):
                for y in range(max(0, int(point[0])-d),min(img.shape[1], int(point[0])+d+1)):
                    dist = exp(-(( point[0] - y) *(point[0] - y)  + (point[1] - x) *(point[1] - x))/32)

                    mask[0][x][y] = 255*dist
                   # mask[0][max(0, int(point[1]) -d): min(img.shape[0], int(point[1])+d+1), max(0, int(point[0]) -d): min(img.shape[1], int(point[0])+d+1)] = 255 / ( dist)


        # cv2.namedWindow("mask[0]", cv2.WINDOW_NORMAL)
        # cv2.imshow("mask[0]", mask[0])
        # cv2.namedWindow("mask[1]", cv2.WINDOW_NORMAL)
        # cv2.imshow("mask[1]", mask[1])

        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)

        mask = torch.FloatTensor(mask)
        mask = ((mask/255 )).float()

        img = Image.fromarray(img.astype(np.uint8))

        img = self.preprocess(img)



        assert -10<= img.max() <= 10
        assert -10<= img.min() <= 10
        #assert 0<= mask.max() <= 1
        #assert 0<= mask.min() <= 1


        return (
            img,
            mask)
