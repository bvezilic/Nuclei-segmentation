import os
import os.path as osp
from glob import glob

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from tqdm import tqdm

from transform import Rescale


class NucleusDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        if self.train:
            self.image_names = os.listdir(osp.join(self.root_dir, "train"))
            self.train_data = []
            self.train_labels = []

            for image_name in tqdm(self.image_names):
                train_img = cv2.imread(osp.join(self.root_dir, "train", image_name, "images", image_name + ".png"))

                self.train_data.append(train_img)

                target_img = np.zeros(train_img.shape[:2], dtype=np.uint8)
                for target in glob(osp.join(self.root_dir, "train", image_name, "masks", "*.png")):
                    target_img_ = cv2.imread(target, 0)
                    target_img = np.maximum(target_img, target_img_)

                self.train_labels.append(target_img)
        else:
            self.image_names = os.listdir(osp.join(self.root_dir, "test"))
            self.test_data = []

            for image_name in tqdm(self.image_names):
                test_img = cv2.imread(osp.join(self.root_dir, "test", image_name, "images", image_name + ".png"))

                self.test_data.append(test_img)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        if self.train:
            image, mask = self.train_data[item], self.train_labels[item]

            if self.transform:
                image = self.transform(image)

            if self.target_transform:
                mask = self.target_transform(mask)

            return image, mask
        else:
            image = self.test_data[item]

            if self.transform:
                image = self.transform(image)

            return image

    def _check_exists(self):
        return osp.exists(osp.join(self.root_dir, "train")) and osp.exists(osp.join(self.root_dir, "test"))


if __name__ == "__main__":
    nucleus_dataset = NucleusDataset(root_dir="./data",
                                     train=True,
                                     transform=Compose([Rescale(256)]),
                                     target_transform=Compose([Rescale(256)]))

    # Display 5 images side-by-side
    for i in range(len(nucleus_dataset)):
        image, mask = nucleus_dataset[i]

        print(i, image.shape, mask.shape)
        cv2.imshow('image', image)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if i == 5:
            break
