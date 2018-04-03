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
    def __int__(self, root_dir, transform=None):
        self.root_dir = root_dir

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, item):
        pass