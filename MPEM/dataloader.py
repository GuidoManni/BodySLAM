'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 04/09/23

Description:
Provide the code for loading the following datasets:
- Internal UCBM dataset
- EndoSlam dataset
'''

# Python standard lib
import sys
import warnings


# AI-lib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Numerical lib
import numpy as np

# Computer Vision lib
import cv2

# Internal module
from UTILS.io_utils import FrameIO


class PoseDatasetLoader(Dataset):
    '''
    Class used by the Pytorch Dataloader to load the dataset
    '''
    def __init__(self, list_of_frames, list_of_absolute_poses = None, dataset_type = None, size_img = 128):
        '''
        Init function

        Parameters:
        - list_of_frames: a list of path [list[str]]
        - list_of_absolute_poses: a list of ground truth poses [list]
        - dataset_type: the dataset to load [str]
        - size_img: new dim of the img [positive int]

        '''
        self.frameIO = FrameIO()
        self.list_of_frames = list_of_frames
        self.list_of_absolute_poses = list_of_absolute_poses
        self.dataset_type = dataset_type
        self.size_img = size_img
        if self.dataset_type == "UCBM":
            self.sequential_transform = transforms.Compose([
                transforms.Resize(self.size_img), # resize the img to the desired size
                transforms.ToTensor(), # convert to tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        elif self.dataset_type == "EndoSlam":
            self.sequential_trans = transforms.Compose([
                transforms.CenterCrop(self.size_img),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        else:
            warnings.warn("[WARNING]: NO Dataset Selected (from UCBM/EndoSlam)")
            sys.exit()

    def __len__(self):
        return len(self.list_of_frames)


    def compute_relative_pose(self, SE3_1, SE3_2):
        '''
        This function computes the relative pose given two poses in SE(3) representation

        Parameters:
        - SE3_1: prev_pose in SE3
        - SE3_2: curr_pose in SE3

        Returns:
        - relative pose between the two
        '''

        # Compute the relative pose in SE(3) representation
        SE3_relative = np.linalg.inv(SE3_1) @ SE3_2

        return SE3_relative

    def __getitem__(self, idx):
        # Check if we are not at the last index
        if self.dataset_type == "EndoSlam":
            if idx < len(self.list_of_frames) - 1:
                frame1 = self.frameIO.load_p_img(self.list_of_frames[idx])
                frame2 = self.frameIO.load_p_img(self.list_of_frames[idx + 1])
                absolute_pose1 = self.list_of_absolute_poses[idx]
                absolute_pose2 = self.list_of_absolute_poses[idx + 1]
            else:
                # If we are at the last index, return the last and second last frames
                frame1 = self.frameIO.load_p_img(self.list_of_frames[-2])  # second last
                frame2 = self.frameIO.load_p_img(self.list_of_frames[-1])  # last
                absolute_pose1 = self.list_of_absolute_poses[-2]  # second last
                absolute_pose2 = self.list_of_absolute_poses[-1]  # last

            # preprocess the input
            input_fr1 = self.sequential_trans(frame1)
            input_fr2 = self.sequential_trans(frame2)

            # get the ground truth poses
            relative_pose = self.compute_relative_pose(absolute_pose1, absolute_pose2)
            relative_pose = torch.tensor(relative_pose)

        elif self.dataset_type == "UCBM":
            # Check if we are not at the last index
            if idx < len(self.list_of_frames) - 1:
                frame1 = self.frameIO.load_p_img(self.list_of_frames[idx])
                frame2 = self.frameIO.load_p_img(self.list_of_frames[idx + 1])
            else:
                # If we are at the last index, return the last and second last frames
                frame1 = self.frameIO.load_p_img(self.list_of_frames[-2])  # second last
                frame2 = self.frameIO.load_p_img(self.list_of_frames[-1])  # last

            relative_pose = -1 # for the UCBM we don't have the ground truth, this is just a place holder

            input_fr1 = self.sequential_trans(frame1)
            input_fr2 = self.sequential_trans(frame2)

        return {"rgb1": input_fr1, "rgb2": input_fr2, "target": relative_pose}