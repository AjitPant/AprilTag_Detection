import os, glob, cv2
import random
from tqdm import tqdm
from PIL import Image
import numpy as np, torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models

class DirDataset(Dataset):

    def __init__(self, img_dir, mask_dir, scale=1):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.scale = scale
        print(self.img_dir)
        try:
            self.ids = (sorted([os.path.splitext(s)[0] for s in os.listdir(self.img_dir)]))
            self.ids.reverse()

        except FileNotFoundError:
            self.ids = []

    def __len__(self):
        return min(3000, len(self.ids))

    def preprocess(self, img, mask=False):

        if not mask:
            trans = transforms.Compose([
                transforms.RandomGrayscale(),
                transforms.ColorJitter(0.3,0.3, 0.3, 0.1 ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # transforms.RandomErasing(p=0.5, scale=(0.001, 0.01), ratio=(0.3, 3.3), value=(0,0,0), inplace=True),
                ])


            img = trans(img).float()

        return img

    def __getitem__(self, i):
        idx = self.ids[i ]
        img_files = [os.path.join(self.img_dir, idx + '.jpg')]
        mask_files = [os.path.join(self.mask_dir, idx + '_0.png'),
                      os.path.join(self.mask_dir, idx + '_1.png'),
                      os.path.join(self.mask_dir, idx + '_2.png'),
                      os.path.join(self.mask_dir, idx + '_3.png'),
                      os.path.join(self.mask_dir, idx + '_4.png'),
                      os.path.join(self.mask_dir, idx + '_5.png'),]


        # assert len(img_files) == 1, f"{idx}: {img_files}"
        # assert len(mask_files) == 2, f"{idx}: {mask_files}"

        img = Image.open(img_files[0])
        mask = [Image.open(mask_file) for mask_file in mask_files]

        if random.random() > 1.5:
            img = transforms.functional.hflip(img)
            real_mask = [];
            for  mas in mask:
                mas = transforms.functional.hflip(mas)
                real_mask.append(mas);
            mask = real_mask

        # Random vertical flipping
        if random.random() > 1.5:
            img = transforms.functional.vflip(img)
            real_mask = [];
            for  mas in mask:
                mas = transforms.functional.vflip(mas)
                real_mask.append(mas);
            mask = real_mask


        # Random affine
        if random.random() > 1.0:
            rotation = random.randint(-180, 180)
            translate = [random.randint(-10,10), random.randint(-10,10)]
            scale = 1.0
            shear = random.randint(-50, 50)

            img = transforms.functional.affine(img, rotation, translate, scale, shear)
            real_mask = [];
            for  mas in mask:
                mas = transforms.functional.affine(mas,  rotation, translate, scale, shear)
                real_mask.append(mas);
            mask = real_mask

        real_mask = [];
        for  mas in mask:
            mas =np.array(mas)
            real_mask.append(mas);
        mask = real_mask


        mask = torch.FloatTensor(mask)
        mask = ((mask /255 ) > 0).float()
        # Random horizontal flipping

        # cv2.imshow("img"+str(i), np.array(img));
        # cv2.imshow("mask"+str(i), mask[0].numpy());
        # cv2.imshow("mask_1"+str(i), mask[1].numpy());
        # cv2.imshow("mask_2"+str(i), mask[2].numpy());
        # cv2.waitKey(0)
        img = self.preprocess(img)

        # cv2.imshow("img"+str(i), np.array(img.cpu().numpy().transpose(1, 2, 0)))
        # cv2.imshow("mask"+str(i), mask[0].cpu().numpy());
        # cv2.imshow("mask_1"+str(i), mask[1].cpu().numpy());
        # cv2.imshow("mask_2"+str(i), mask[2].cpu().numpy());
        # cv2.waitKey(0)
        # print(mask.shape)
        # print(img.shape)

        # if random.random() > 0.5:
        #     img = transforms.functional.hflip(img)
        #     mask = transforms.functional.hflip(mask)

        # # Random vertical flipping
        # if random.random() > 0.5:
        #     img = transforms.functional.vflip(img)
        #     mask = transforms.functional.vflip(mask)



        return (
            img,
            mask)
