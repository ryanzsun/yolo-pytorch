import argparse
import datetime
import os
import sys
import time

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from darknet import *
from utils.pascal_dataset import *
from utils.utils import *
from yolo_loss import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--weights_path", type=str, default=None, help="path to weights file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints", help="directory where model checkpoints are saved"
)
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
parser.add_argument("--learning_rate", type=float, default= 3e-4, help="learning rate")
parser.add_argument("--n_classes", type=float, default= 20, help="number of classes")
parser.add_argument("--n_anchors", type=float, default= 3, help="number of anchors")

opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available() and opt.use_cuda


train_data = VOCDataset(test=False)
test_data = VOCDataset(test=True)
train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

if opt.weights_path is not None:
    model = torch.load(opt.weights_path, map_location=lambda storage, loc: storage)
else:
    model = Darknet(3, opt.n_classes, opt.n_anchors, ResidualBlock, [1,2,8,8,4])
    model.apply(init_weights)

model.train()

optimizer = AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=0.0005)
scheduler = CyclicLRWithRestarts(optimizer, opt.batch_size, opt.epochs, restart_period=10, t_mult=2, policy="cosine")

