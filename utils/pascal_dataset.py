import glob

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self, image_size = 416, test = False):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('JPEGImages', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.is_test = test
        self.img_shape = (img_size, img_size)

    def __getitem__(self, index):
        