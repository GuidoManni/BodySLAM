'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Created on: 25/09/23

Description:
This script provides the interface to the Monocular Pose Estimation Module 
'''

# Python Standard Lib
import warnings

# AI-lib 
import torch
from torchvision import transforms

# Internal Modules
from UTILS.io_utils import FrameIO, ModelIO
from MPEM.architecture_v3 import *
# Computational Lib
import numpy as np

class MPEMInterface:
    # This class call/initialize the instances to interface with the MPE

    def __init__(self, path_to_model):

        # step 2: call other classes here
        self.frameIO = FrameIO()
        self.modelIO = ModelIO()

        # put constant here:
        self.input_shape = (6, 256, 256)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # step 1: load the model
        self.pose_model = self._initialize_pose_model(path_to_model).eval()

        # trasformation
        self.sequential_transform_with_crop = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.sequential_transform_resize = transforms.Compose([
            transforms.Resize(128),  # resize the img to the desired size
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])


    def _initialize_pose_model(self, path_to_model):
        '''
        This function load the pose model in order to perform inference
        :param path_to_model: the path to the model
        :return pose_model: the torch model
        '''

        # step 1: we need to load the pose model
        print(f"[INFO] model loaded on {self.device}")
        GAB = ConditionalGenerator(input_shape=self.input_shape, device = self.device).to(self.device)

        # step 2: we load the weigths of the model
        GAB, _, _ = self.modelIO.load_pose_model(path_to_model, GAB)

        return GAB

    def infer_relative_pose_between(self, path_frame1, path_frame2, type_of_trans = 'crop'):
        '''
        This function uses the Monocular Pose Module to infer the relative pose between to consecutive frames
        :param frame1: the prev frame
        :param frame2: the curr frame
        :param type_of_trans: the type of transformation to be used. Can be 'crop' or 'resize'
        :return: pose matrix in SE(3)
        '''

        assert type_of_trans == 'crop' or type_of_trans == 'resize', "type_of_trans must be 'crop' or 'resize'!"

        # first we load the images
        frame1 = self.frameIO.load_p_img(path_frame1)
        frame2 = self.frameIO.load_p_img(path_frame2)

        # then we perform the same transformation we used during training
        if type_of_trans == 'crop':
            frame1 = self.sequential_transform_with_crop(frame1).to(self.device)
            frame2 = self.sequential_transform_with_crop(frame2).to(self.device)
        elif type_of_trans == 'resize':
            frame1 = self.sequential_transform_resize(frame1).to(self.device)
            frame2 = self.sequential_transform_resize(frame2).to(self.device)

        frame1 = frame1.unsqueeze(0)
        frame2 = frame2.unsqueeze(0)
        frame12 = torch.cat([frame1, frame2], dim=1)
        with torch.no_grad():
            # let's infer the pose
            motion_matrix_SE3 = self.pose_model(frame12, mode='pose')

        return motion_matrix_SE3.squeeze().cpu().numpy()







        

