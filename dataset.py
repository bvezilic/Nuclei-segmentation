import os
import os.path as osp
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class NucleusDataset(Dataset):
    def __int__(self, root_dir, train=True, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

        if self.train:
            self.train_data = None
        else:
            self.test_data = None

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, item):
        if self.train:
            img, target = self.train_data[item], self.train_labels[item]
        else:
            img, target = self.test_data[item], self.test_labels[item]


    def _check_exists(self):
        return osp.exists(osp.join(self.root_dir, "train")) and osp.exists(osp.join(self.root_dir, "test"))

