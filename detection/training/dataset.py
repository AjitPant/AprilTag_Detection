import os, glob, cv2
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
            self.ids = [s.split('.')[0] for s in os.listdir(self.img_dir)]
        except FileNotFoundError:
            self.ids = []

    def __len__(self):
        return len(self.ids)

    def preprocess(self, img, mask=False):
        w, h = img.size
        _h = int(h * self.scale)
        _w = int(w * self.scale)

        assert _w > 0
        assert _h > 0

        _img = img.resize((_w, _h))
        _img = np.array(_img)
        if len(_img.shape) == 2:
            _img = np.expand_dims(_img, axis=(-1))

        if _img.max()>1  and not mask:
            _img = _img/255

        if not mask:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            _img = trans(_img).float()

        return _img

    def __getitem__(self, i):
        idx = self.ids[i]
        img_files = glob.glob(os.path.join(self.img_dir, idx + '.*'))
        mask_files = sorted(glob.glob(os.path.join(self.mask_dir, idx + '_*.*')))

        assert len(img_files) == 1, f"{idx}: {img_files}"
        assert len(mask_files) == 2, f"{idx}: {mask_files}"

        img = Image.open(img_files[0])
        mask = [cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) for mask_file in mask_files]

        mask[0]=mask[0]/255.0

        img = self.preprocess(img)
        return (
         img,
         torch.from_numpy(np.array(mask)).float())
