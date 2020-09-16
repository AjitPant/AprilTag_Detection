import os, glob, cv2
import random
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
        original_height = 512
        original_width = 512


        self.aug = A.Compose([
            A.OneOf([
                        A.ElasticTransform(),
                        A.GridDistortion(),
                        A.OpticalDistortion(),
                    ], p=0.1),

            A.OneOf([
                A.RandomSizedCrop(min_max_height=(original_height//8, original_height),
                                  height=original_height, width=original_width, p=0.5),
                # A.RandomSizedCrop(min_max_height=(original_height//3, original_height),
                #                   height=original_height//2, width=original_width//2, p=0.5),
                A.PadIfNeeded(min_height=original_height,
                              min_width=original_width, p=0.5)
            ], p=0.5),
            A.OneOf([
                A.Blur(),
                A.MotionBlur(),
            ], p=0.1),

            A.OneOf([
                A.ToGray(),
                A.CoarseDropout(max_height=10, min_height=2,
                                max_width=10, min_width=2, min_holes=1),
                A.ChannelDropout(),
            ], p=0.5),


            A.OneOf([
                A.VerticalFlip(p=0.9),
                A.HorizontalFlip(p=0.9),
            ], p=1.0),

            A.OneOf([
                A.RandomRotate90(p=0.9),
                A.Rotate(limit=180, p=0.9),
            ], p=1.0),
            # A.CLAHE(p=0.8),
            A.GaussNoise(),
            A.OneOf([
                A.RandomSnow(),
                A.RandomRain(),
                A.RandomFog(),
            ], p=0.0),
            A.OneOf([
                A.RandomShadow(p=0.8),
                A.RandomSunFlare(src_radius=40),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RGBShift(p=0.5),
                A.RandomGamma(p=0.8)
            ], p=0.1),
        ], p=0.8,
            additional_targets={
            'image': 'image',

            'mask0': 'mask',
            'mask1': 'mask',
        })

        print(self.img_dir)
        try:
            self.ids = (sorted([os.path.splitext(s)[0]
                                for s in os.listdir(self.img_dir)]))
            # self.ids.reverse()

        except FileNotFoundError:
            self.ids = []

    def __len__(self):
        return min(2000, len(self.ids))

    def preprocess(self, img, mask=False):

        if not mask:
            trans = transforms.Compose([
                # transforms.RandomGrayscale(),
                # transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
                transforms.ToTensor(),
                # transforms.RandomErasing(p=0.1, scale=(0.001, 0.05), ratio=(
                #     0.3, 3.3), value=(0, 0, 0), inplace=True),
                # # transforms.RandomErasing(p=0.2, scale=(0.001, 0.05), ratio=(
                # #     0.3, 3.3), value=(10, 10, 10), inplace=True),
                # transforms.RandomErasing(p=0.3, scale=(0.001, 0.05), ratio=(
                #     0.3, 3.3), value=(0, 30, 0), inplace=True),
                # transforms.RandomErasing(p=0.4, scale=(0.001, 0.05), ratio=(
                #     0.3, 3.3), value=(50, 0, 200), inplace=True),
                # transforms.RandomErasing(p=0.4, scale=(0.001, 0.05), ratio=(
                # 0.3, 3.3), value=(100, 20, 0), inplace=True),
                # transforms.RandomErasing(p=0.4, scale=(0.001, 0.05), ratio=(
                #     0.3, 3.3), value=(13, 50, 0), inplace=True),
                # transforms.RandomErasing(p=0.4, scale=(0.001, 0.05), ratio=(
                #     0.3, 3.3), value=(15, 58, 10), inplace=True),
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

        assert len(img_files) == 1, f"{idx}: {img_files}"
        assert len(mask_files) == 2, f"{idx}: {mask_files}"

        img = cv2.imread(img_files[0])
        mask = [cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                for mask_file in mask_files]
        # mask[0] = (mask[0] >0).astype(np.uint8)*255
        # mask[1] = (mask[1] >0).astype(np.uint8)*255
        augmented = self.aug(image=img, mask0=mask[0], mask1=mask[1])
        img = augmented['image']
        mask[0] = augmented['mask0']
        mask[1] = augmented['mask1']

        mask = torch.FloatTensor(mask)
        mask = ((mask / 255) > 0.25).float()

        img = Image.fromarray(img)
        # print(img.shape)
        img = self.preprocess(img)

        # cv2.imshow("img"+str(i), img.numpy().transpose((1, 2, 0)))
        # # cv2.imshow("mask"+str(i), mask[0].numpy())
        # # cv2.imshow("mask_1"+str(i), mask[1].numpy())
        # cv2.waitKey(0)

        # if random.random() > 0.5:
        #     img = transforms.functional.hflip(img)
        #     real_mask = [];
        #     for  mas in mask:
        #         mas = transforms.functional.hflip(mas)
        #         real_mask.append(mas);
        #     mask = real_mask

        # # Random vertical flipping
        # if random.random() > 0.5:
        #     img = transforms.functional.vflip(img)
        #     real_mask = [];
        #     for  mas in mask:
        #         mas = transforms.functional.vflip(mas)
        #         real_mask.append(mas);
        #     mask = real_mask

        # # Random affine
        # if random.random() > 0.5:
        #     rotation = random.randint(-180, 180)
        #     translate = [random.randint(-100,100), random.randint(-100,100)]
        #     scale = 1.0
        #     shear = random.randint(-50, 50)

        #     img = transforms.functional.affine(img, rotation, translate, scale, shear)
        #     real_mask = [];
        #     for  mas in mask:
        #         mas = transforms.functional.affine(mas,  rotation, translate, scale, shear)
        #         real_mask.append(mas);
        #     mask = real_mask

        # real_mask = [];
        # for  mas in mask:
        #     mas =np.array(mas)
        #     real_mask.append(mas);
        # mask = real_mask

        # # Random horizontal flipping

        # # cv2.imshow("img"+str(i), np.array(img.cpu().numpy().transpose(2, 1, 0)))
        # # cv2.imshow("mask"+str(i), mask[0].cpu().numpy());
        # # cv2.imshow("mask_1"+str(i), mask[1].cpu().numpy());
        # # cv2.imshow("mask_2"+str(i), mask[2].cpu().numpy());
        # # cv2.waitKey(0)
        # # print(mask.shape)
        # # print(img.shape)

        # # if random.random() > 0.5:
        # #     img = transforms.functional.hflip(img)
        # #     mask = transforms.functional.hflip(mask)

        # # # Random vertical flipping
        # # if random.random() > 0.5:
        # #     img = transforms.functional.vflip(img)
        # #     mask = transforms.functional.vflip(mask)

        return (
            img,
            mask)
