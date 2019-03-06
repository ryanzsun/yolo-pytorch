import glob
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")

class VOCDataset(Dataset):
    def __init__(self, img_size = 416, test = False):
        if not test:
            list_path = "/media/ryan/hdd/Data/PascalVOC/VOC2012/2012_trainval.txt"
        else:
            list_path = "/media/ryan/hdd/Data/PascalVOC/VOC2007/2007_trainval.txt"
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('JPEGImages', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.is_test = test
        self.img_shape = (img_size, img_size)
        self.max_objects = 50

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # load image
        img_path = self.img_files[index].rstrip()
        img = np.array(Image.open(img_path))
        if len(img.shape)!=3:
            img = np.stack((img,)*3, axis = -1)
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        input_img = np.pad(img, pad, 'constant', constant_values=128)

        padded_h, padded_w, _ = input_img.shape
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        input_img = np.transpose(input_img, (2, 0, 1))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        input_img = torch.from_numpy(input_img).float()
        input_img = normalize(input_img)


        #load label
        label_path = self.label_files[index].rstrip()
        print(label_path)
        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)

            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            print(x1,y1,x2,y2)

            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h

        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)

        return img_path, input_img, filled_labels


if __name__ == "__main__":
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    train_data = VOCDataset()
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, num_workers=1)
    for idx, data in enumerate(train_loader):

        pass
