'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 04/09/23


Description:
Provide the function used to load and save images, files, etc...
'''
# Python standard-lib
import warnings
import csv
import os

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


    def load_cv2_depth(self, path_to_depth):
        '''
        This function load a depth map using opencv
        :param path_to_depth: path to the depth map
        :return: depth map
        '''
        depth_map = cv2.imread(path_to_depth, cv2.IMREAD_ANYDEPTH)

        return depth_map



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


class CSVIO:
    def write_metrics_on_cvs(self, saving_path, metrics):
        headers = []
        if 'avg' in saving_path:
            for keys in metrics.keys():
                headers.append(keys)

            with open(saving_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                writer.writerow(metrics)
        else:
            for keys in metrics[0].keys():
                headers.append(keys)

            with open(saving_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headers)
                writer.writeheader()
                for i in range(len(metrics)):
                    writer.writerow(metrics[i])




class DatasetLoader:
    def read_Hamlyn(self, path_to_Hamlyn):
        '''
        This function obtain the full relative path of all the images.
        :param path_to_Hamlyn: path to the dataset
        :return: dict_of_path
        '''
        '''
        To use this function the directory must follow this structure!
        Directory Structure:
        Hamlyn
        -> rectified01
            -> image01
            -> image02
            -> depth01
            -> depth02
        ...
        -> rectified27
        '''

        dataset_paths = {}

        # step 1: we read the content of the root folder
        hamlyn_root_content = os.listdir(path_to_Hamlyn)
        print("[INFO]: found the following content in the folder provided")
        print(hamlyn_root_content)

        # step 2: we remove in the list any eventual folder that is not "rectified"
        print("[INFO]: removing any folder that is not 'rectified'")
        hamlyn_root_content_path = []
        for i in range(len(hamlyn_root_content)):
            if "rectified" in hamlyn_root_content[i]:
                hamlyn_root_content_path.append(os.path.join(path_to_Hamlyn, hamlyn_root_content[i]))
        hamlyn_root_content_path = sorted(hamlyn_root_content_path)

        # step 3: we extract the depth & images
        for i in range(len(hamlyn_root_content_path)):
            rectified_content = os.listdir(hamlyn_root_content_path[i])
            folder_name = hamlyn_root_content_path[i].split("/")[-1]
            print(f"[INFO]: loading file from {hamlyn_root_content_path[i]}")
            rectified_content_dict = {}
            for j in range(len(rectified_content)):
                list_content = []
                if rectified_content[j] == "depth01":
                    tmp = os.listdir(os.path.join(hamlyn_root_content_path[i], rectified_content[j]))
                    for elem in tmp:
                        if '.png' in elem:
                            list_content.append(os.path.join(hamlyn_root_content_path[i], rectified_content[j], elem))
                    rectified_content_dict["depth01"] = sorted(list_content)

                elif rectified_content[j] == "depth02":
                    tmp = os.listdir(os.path.join(hamlyn_root_content_path[i], rectified_content[j]))
                    for elem in tmp:
                        if '.png' in elem:
                            list_content.append(os.path.join(hamlyn_root_content_path[i], rectified_content[j], elem))
                    rectified_content_dict["depth02"] = sorted(list_content)

                elif rectified_content[j] == "image01":
                    tmp = os.listdir(os.path.join(hamlyn_root_content_path[i], rectified_content[j]))
                    for elem in tmp:
                        if '.jpg' in elem:
                            list_content.append(os.path.join(hamlyn_root_content_path[i], rectified_content[j], elem))
                    rectified_content_dict["image01"] = sorted(list_content)
                elif rectified_content[j] == "image02":
                    tmp = os.listdir(os.path.join(hamlyn_root_content_path[i], rectified_content[j]))
                    for elem in tmp:
                        if '.png' in elem:
                            list_content.append(os.path.join(hamlyn_root_content_path[i], rectified_content[j], elem))
                    rectified_content_dict["image02"] = sorted(list_content)

                dataset_paths[folder_name] = rectified_content_dict

        return dataset_paths

    def read_EndoSlam(self, path_to_EndoSlam):
        '''
        This function read the EndoSlam dataset
        :param path_to_EndoSlam:
        :return: dict of paths
        '''

        '''
        This function works only with the standard folder architecture of EndoSlam
        EndoSlam
        -> 3D Scanners
        -> Cameras
        -> OlympusCam
        -> UnityCam
            -> Calibration
            -> Colon
                -> Frames
                -> Pixelwise Depths
                -> Poses
            -> Small Intestine
                -> Frames
                -> Pixelwise Depths
                -> Poses
            -> Stomach
                -> Frames
                -> Pixelwise Depths
                -> Poses
        '''
        FOLDERS = ['Frames', 'Pixelwise Depths']
        dataset_paths = {}

        # step 1: navigate to UnityCam
        if "UnityCam" not in path_to_EndoSlam:
            # if the folder is not in the path we need to add it
            path_to_EndoSlam = os.path.join(path_to_EndoSlam, "UnityCam")

        # step 2: get the root content of the UnityCam and build the relative path
        content_unitycam = os.listdir(path_to_EndoSlam)
        path_to_content_unitycam = []
        for file in content_unitycam:
            if file != "Calibration":  # we add only the folders of interest
                path_to_content_unitycam.append(os.path.join(path_to_EndoSlam, file))

        # step 3: now we get the full relative path of all the content inside UnityCam
        for i in range(len(path_to_content_unitycam)):
            dict_key = path_to_content_unitycam[i].split("/")[-1]
            path_inside_folder = {}
            for folder in FOLDERS:
                paths = []
                intermediate_path = os.path.join(path_to_content_unitycam[i], folder)
                content_folder = sorted(os.listdir(intermediate_path))
                for content in content_folder:
                    path = os.path.join(intermediate_path, content)
                    paths.append(path)
                path_inside_folder[folder] = paths
            dataset_paths[dict_key] = path_inside_folder

        return dataset_paths
