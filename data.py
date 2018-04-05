import os
import os.path as osp
import torch
from glob import glob
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from tqdm import tqdm


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
                train_img = io.imread(osp.join(self.root_dir, "train", image_name, "images", image_name + ".png"))
                train_img = train_img[:, :, 3]

                if self.transform:
                    train_img = transform(train_img)
                self.train_data.append(train_img)

                target_img = np.zeros(train_img.shape[:-1], dtype=np.bool)
                for target in glob(osp.join(self.root_dir, "train", image_name, "masks", "*.png")):
                    target_img_ = io.imread(target, as_grey=True)
                    target_img = np.maximum(target_img, target_img_)

                if self.target_transform:
                    target_img = target_transform(target_img)
                self.train_labels.append(target_img)
        else:
            self.image_names = os.listdir(osp.join(self.root_dir, "test"))
            self.test_data = []
            for image_name in tqdm(self.image_names):
                test_img = io.imread(osp.join(self.root_dir, "test", image_name, "images", image_name + ".png"))
                test_img = test_img[:, :, 3]

                if self.transform:
                    test_img = transform(test_img)
                self.train_data.append(test_img)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        if self.train:
            img, mask = self.train_data[item], self.train_labels[item]
            return {"image": img, "mask": mask}
        else:
            img = self.test_data[item]
            return {"image": img}

    def _check_exists(self):
        return osp.exists(osp.join(self.root_dir, "train")) and osp.exists(osp.join(self.root_dir, "test"))


class Rescale:
    def __init__(self):
        pass

    def __call__(self, sample):
        pass


if __name__ == "__main__":
    nucleus_dataset = NucleusDataset(root_dir="data", train=True)

    for i in range(len(nucleus_dataset)):
        sample = nucleus_dataset[i]

        print(i, sample.get('image').shape, sample.get('mask').shape)

        io.imshow(sample.get('image'))
        io.imshow(sample.get('mask'))
