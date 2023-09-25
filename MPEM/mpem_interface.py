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

# Internal Modules
from UTILS.io_utils import FrameIO 

class MPEMInterface:
    # This class call/initialize the instances to interface with the MPE

    def __init__(self, path_to_model):
        # step 1: load the model
        self.pose_model = self._initialize_pose_model(path_to_model)
        # step 2: call other classes here
        self.frameIO = FrameIO()
    
    def _initialize_pose_model(self, path_to_model,):
        '''
        This function load the pose model in order to perform inference
        :param path_to_model: the path to the model
        :return pose_model: the torch model
        '''

        

