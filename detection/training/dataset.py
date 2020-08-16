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
        return min(8000, len(self.ids))

    def preprocess(self, img, mask=False):

        if not mask:
            trans = transforms.Compose([
                transforms.ColorJitter(0.5,0.5, 0.5, 0.2 ),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=(0,0,0), inplace=True),
                ])


            img = trans(img).float()

        return img

    def __getitem__(self, i):
        idx = self.ids[i]
        img_files = [os.path.join(self.img_dir, idx + '.jpg')]
        mask_files = [os.path.join(self.mask_dir, idx + '_1.png'),
                      os.path.join(self.mask_dir, idx + '_2.png')]


        # assert len(img_files) == 1, f"{idx}: {img_files}"
        # assert len(mask_files) == 2, f"{idx}: {mask_files}"

        img = Image.open(img_files[0])
        mask = [cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) for mask_file in mask_files]

        mask[0]=mask[0]/255.0

        mask = torch.FloatTensor(mask);
        # Random horizontal flipping

        img = self.preprocess(img)

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
