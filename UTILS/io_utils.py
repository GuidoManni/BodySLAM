'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 04/09/23


Description:
Provide the function used to load and save images, files, etc...
'''
# Python standard-lib
import warnings

# Computer Vision lib
from PIL import Image
import cv2

# AI-lib
import torch
from torch.utils.data import DataLoader

# Numerical lib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# Internal module
from UTILS.geometry_utils import LieEuclideanMapper

class FrameIO:
    def load_p_img(self, path_to_img, convert_to_rgb = False):
        '''
        This function load an img using the Image from PIL

        Parameters:
        - path_to_img: path of the img [str]
        - convert_to_rgb: Flag value [bool]

        Return:
        - PIL image
        '''
        if convert_to_rgb:
            return Image.open(path_to_img).convert("RGB")
        else:
            return Image.open(path_to_img)

    def save_p_img(self, img, saving_path, extension = None):
        '''
        This function save a PIL img

        Parameters:
        - img: image to be saved [PIL image]
        - saving_path: path for saving the img [str]
        - extension: the extension of the img

        Return:
        - True (if success)
        - False (if failed)
        '''
        try:
            if extension is None:
                # no extension has been provided, this assumes that it is in the saving path
                warnings.warn("No extension has been provided")
                img.save(saving_path)
                return True
            else:
                saving_path_with_extension = saving_path + extension
                img.save(saving_path_with_extension)
                return True
        except ValueError as e:
            print(f"Error while saving: {e}")


    def load_cv2_img(self, path_to_img, convert_to_rgb = False):
        '''
        This function load an img using the opencv-python lib

        Parameters:
        - path_to_img: path of the img [str]
        - convert_to_rgb: Flag value [bool]

        Return:
        - cv2 img
        '''
        image = cv2.imread(path_to_img)

        if convert_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def save_cv2_img(self, img, saving_path, extension = None):
        '''
        This function save a cv2 img

        Parameters:
        - img: image to be saved [cv2 image]
        - saving_path: path for saving the img [str]
        - extension: the extension of the img

        Return:
        - True (if success)
        - False (if failed)
        '''
        try:
            if extension is None:
                # no extension has been provided, this assumes that it is in the saving path
                warnings.warn("No extension has been provided")
                cv2.imwrite(saving_path, img)
                return True
            else:
                saving_path_with_extension = saving_path + extension
                cv2.imwrite(saving_path_with_extension, img)
                return True
        except ValueError as e:
            print(f"Error while saving: {e}")



class XlsxIO:
    def read_xlsx_pose_file(self, filepath, convert_to_se3 = False):
        '''
        read the ground truth poses stored in a XLSX files using pandas and conver them in se(3) (LIE space)

        Parameter:
        - filepath: the file path to the xlsx

        Return:
        - motion matrices
        '''
        # Initialize an empty list to store transformation matrices
        motion_matrices = []

        df = pd.read_excel(filepath)

        for i, row in df.iterrows():
            # Extract translation
            translation_vector = np.array([row['trans_x'], row['trans_y'], row['trans_z']])

            # Extract rotation
            rotation_quaternion = np.array([row['quot_x'], row['quot_y'], row['quot_z'], row['quot_w']])

            # Convert quaternion rotation to a rotation matrix
            rotation = R.from_quat(rotation_quaternion)
            rotation_matrix = rotation.as_matrix()

            # Construct the 4x4 transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[0:3, 0:3] = rotation_matrix
            transformation_matrix[0:3, 3] = translation_vector

            if convert_to_se3:
                # Convert to lie space for training purposes
                pose_se3 = LieEuclideanMapper.SE3_to_se3(transformation_matrix)

                motion_matrices.append(pose_se3)
            else:
                motion_matrices.append(transformation_matrix)

        return motion_matrices

class ModelIO:
    def load_pose_model(self, path_to_the_model, model, optimizer):
        '''
        This function load the pose model.

        Parameters:
        - path_to_the_model: the path to the model [str]
        - model: pytorch model
        - optimizer: pytorch optimizer

        Return
        - model: the loaded model
        - optimizer: the loaded optimizer
        - training_var: dictionary containing the epoch, iter_on_ucbm and the best loss
        '''

        checkpoint = torch.load(path_to_the_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        training_var = {'epoch': checkpoint['epoch'],
                        'iter_on_ucbm': checkpoint['iter_on_ucbm'],
                        'ate': checkpoint['ate'],
                        'are': checkpoint['are'],
                        'rte': checkpoint['rte'],
                        'rre': checkpoint['rre']}
        return model, optimizer, training_var

    def save_pose_model(self, saving_path, model, optimizer, training_var, best_model):
        '''
        This function save the pose model

        Parameters:
        - saving_path: the path where to save the model [str]
        - model: the model to save [pytorch model]
        - optimizer: the optimizer to save [pytorch optimizer]
        - training_var: dict containing some training variable to save [dict]
        - best_model: flag value to tell if it's the best model or not [bool]
        '''

        if best_model:
            curr_name = saving_path.split("/")[-1]
            new_name = curr_name.replace("model", "best_model")
            saving_path = saving_path.replace(curr_name, new_name)
        torch.save({
            'epoch': training_var['epoch'],
            'iter_on_ucbm': training_var['iter_on_ucbm'],
            'ate': training_var['ate'],
            'are': training_var['are'],
            'rte': training_var['rte'],
            'rre': training_var['rre'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, saving_path)





