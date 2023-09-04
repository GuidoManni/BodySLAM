'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 04/09/23

Description:
It's the train script used to train the pose network
'''


# Python standard lib
import sys
import os
import itertools

# Numerical lib
import numpy as np

# Computer Vision lib
import cv2

# AI-lib
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

# Stat lib
import wandb

# Internal Module



