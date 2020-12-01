import os, glob, cv2
import random
import pickle
from tqdm import tqdm
from PIL import Image
import numpy as np, torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
import albumentations as A

class DirDataset(Dataset):

    def __init__(self, img_dir, mask_dir, scale=1):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.scale = scale
        original_height = 512*2
        original_width = 512*2


        self.aug = A.Compose([

            A.OneOf([
                A.RandomSizedCrop(min_max_height=(original_height//4, original_height),
                                  height=original_height, width=original_width, p=0.5),
                A.PadIfNeeded(min_height=original_height,
                              min_width=original_width, p=0.5)
            ], p=0.5),
            A.OneOf([
                A.Blur((5,11), p = 0.5),
                A.MotionBlur((5,11),p =  0.5),
            ], p=0.5),

            A.OneOf([
                A.ToGray(),
                # A.CoarseDropout(max_height=40, min_height=2,
                #                 max_width=40, min_width=2, min_holes=1, max_holes=20,fill_value=(255,0,255)),
                A.ChannelDropout(),
            ], p=0.5),


            A.OneOf([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
            ], p=0.1),

            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=180, p=0.5),
            ], p=0.5),
            A.CLAHE(p=0.3),
            A.GaussNoise(),
            A.OneOf([
                # A.RandomSnow(p=1.0),
                A.RandomRain(),
                A.RandomFog( fog_coef_lower = 0.1, fog_coef_upper = 0.3),
            ], p=0.1),
            A.OneOf([
                A.RandomShadow(p=0.8, num_shadows_upper=5),
                A.RandomSunFlare(src_radius=40),
            ], p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RGBShift(p=0.5),
                A.RandomGamma(p=0.8)
            ], p=0.4),
        ], p=0.8,
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
        with open(keypoints_file, "rb") as f:
            keypoints = np.array(pickle.load(f)).reshape((-1,2)).tolist()
        augmented = self.aug(image=img, mask0=mask[0], mask1=mask[1], keypoints = keypoints)
        img = augmented['image']

        mask[1] = augmented['mask1']
        mask[0] = augmented['mask0']

        keypoints = augmented['keypoints']

        mask[0].fill(0)

        d = 1

        for point in keypoints:
            mask[0][max(0, int(point[1]) -d): min(img.shape[0], int(point[1])+d+1), max(0, int(point[0]) -d): min(img.shape[1], int(point[0])+d+1)] = 255





        mask = torch.FloatTensor(mask)
        mask = ((mask / 255)).float()

        img = Image.fromarray(img)

        img = self.preprocess(img)

        return (
            img,
            mask)
