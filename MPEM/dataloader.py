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
import os
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
from UTILS.io_utils import FrameIO, XlsxIO
from UTILS.geometry_utils import PoseOperator as PO


class PoseDatasetLoader(Dataset):
    '''
    Class used by the Pytorch Dataloader to load the dataset
    '''
    def __init__(self, list_of_frames, list_of_depth, list_of_absolute_poses = None, dataset_type = None, size_img = 128):
        '''
        Init function

        Parameters:
        - list_of_frames: a list of path [list[str]]
        - list_of_absolute_poses: a list of ground truth poses [list]
        - dataset_type: the dataset to load [str]
        - size_img: new dim of the img [positive int]

        '''
        self.frameIO = FrameIO()
        self.list_of_frames = sorted(list_of_frames)
        self.list_of_depths = sorted(list_of_depth)
        self.list_of_absolute_poses = list_of_absolute_poses
        self.dataset_type = dataset_type
        self.size_img = size_img
        self.sequential_transform_dp = transforms.Compose([
            transforms.Resize(self.size_img),
            transforms.Lambda(self.depth_map_to_tensor),
        ])
        if self.dataset_type == "UCBM":
            self.sequential_transform = transforms.Compose([
                transforms.Resize(self.size_img), # resize the img to the desired size
                transforms.ToTensor(), # convert to tensor
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        elif self.dataset_type == "EndoSlam":
            self.sequential_transform = transforms.Compose([
                transforms.CenterCrop(self.size_img),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

        else:
            warnings.warn("[WARNING]: NO Dataset Selected (from UCBM/EndoSlam)")
            sys.exit()

    def depth_map_to_tensor(self, depth_map):
        # Convert the loaded depth map into a numpy array of type float32
        depth_array = np.array(depth_map, dtype=np.float32)

        # Normalize to a range between 0 and 1
        min_val = np.min(depth_array)
        max_val = np.max(depth_array)
        normalized_depth_array = (depth_array - min_val) / (
                    max_val - min_val + 1e-7)  # Adding a small value to prevent division by zero

        # Convert the normalized numpy.float32 array into a PyTorch tensor
        return torch.tensor(normalized_depth_array)

    def __len__(self):
        return len(self.list_of_frames)


    def __getitem__(self, idx):
        # Check if we are not at the last index
        if self.dataset_type == "EndoSlam":
            if idx < len(self.list_of_frames) - 1:
                frame1 = self.frameIO.load_p_img(self.list_of_frames[idx])
                frame2 = self.frameIO.load_p_img(self.list_of_frames[idx + 1])
                dp1 = self.frameIO.load_p_img(self.list_of_depths[idx])
                dp2 = self.frameIO.load_p_img(self.list_of_depths[idx + 1])
                absolute_pose1 = self.list_of_absolute_poses[idx]
                absolute_pose2 = self.list_of_absolute_poses[idx + 1]
            else:
                # If we are at the last index, return the last and second last frames
                frame1 = self.frameIO.load_p_img(self.list_of_frames[-2])  # second last
                frame2 = self.frameIO.load_p_img(self.list_of_frames[-1])  # last
                dp1 = self.frameIO.load_p_img(self.list_of_depths[-2])
                dp2 = self.frameIO.load_p_img(self.list_of_depths[-1])
                absolute_pose1 = self.list_of_absolute_poses[-2]  # second last
                absolute_pose2 = self.list_of_absolute_poses[-1]  # last

            # preprocess the input
            input_fr1 = self.sequential_transform(frame1)
            input_fr2 = self.sequential_transform(frame2)
            input_dp1 = self.sequential_transform_dp(dp1)
            input_dp2 = self.sequential_transform_dp(dp2)

            # get the ground truth poses
            relative_pose = PO.compute_relative_pose(absolute_pose1, absolute_pose2)
            relative_pose = torch.tensor(relative_pose)
            target = (torch.tensor(absolute_pose1), torch.tensor(absolute_pose2), relative_pose)

        elif self.dataset_type == "UCBM":
            # Check if we are not at the last index
            if idx < len(self.list_of_frames) - 1:
                frame1 = self.frameIO.load_p_img(self.list_of_frames[idx])
                frame2 = self.frameIO.load_p_img(self.list_of_frames[idx + 1])
                dp1 = self.frameIO.load_p_img(self.list_of_depths[idx])
                dp2 = self.frameIO.load_p_img(self.list_of_depths[idx + 1])
            else:
                # If we are at the last index, return the last and second last frames
                frame1 = self.frameIO.load_p_img(self.list_of_frames[-2])  # second last
                frame2 = self.frameIO.load_p_img(self.list_of_frames[-1])  # last
                dp1 = self.frameIO.load_p_img(self.list_of_depths[-2])
                dp2 = self.frameIO.load_p_img(self.list_of_depths[-1])

            target = -1 # for the UCBM we don't have the ground truth, this is just a place holder

            input_fr1 = self.sequential_transform(frame1)
            input_fr2 = self.sequential_transform(frame2)
            input_dp1 = self.sequential_transform_dp(dp1)
            input_dp2 = self.sequential_transform_dp(dp2)



        return {"rgb1": input_fr1, "rgb2": input_fr2, "dp1": input_dp1, "dp2": input_dp2, "target": target}

class DatasetsIO:
    def __init__(self):
        self.xlsxIO = XlsxIO()

    def list_dir_with_relative_path_to(self, path, sort=True, mode='folder', extension=None, filter_by_word=None, revert_filter=False):
        '''
        Exctract the full path of a directory
        Parameters
        - path -> path to the folder [str]
        - sort -> True if we want to sort the result, False otherwise [bool]
        - mode -> can be 'folder' [default value] or 'file' [str]
        - extension -> None (if we want to extract all types of files) otherwise str (the extension))
        - revert_filter -> if true it will NOT extract the files that has contains the filter_word

        Return
        - list_of_abs_path -> list obj
        '''
        # inzializziamo le variabili
        list_of_abs_path = []

        # step 1: otteniamo il contenuto del path fornito
        root_content = os.listdir(path)

        if sort:
            root_content = sorted(root_content)

        # step 2: costruiamo il percorso assoluto
        for content in root_content:
            tmp = os.path.join(path, content)
            if mode == 'folder':
                # se stiamo raccogliendo solo cartella allora verifichiamo che esse siano cartelle
                if os.path.isdir(tmp):
                    if filter_by_word is None:
                        list_of_abs_path.append(tmp)
                    else:
                        if not revert_filter:
                            if filter_by_word in tmp:
                                list_of_abs_path.append(tmp)
                        elif revert_filter:
                            if filter_by_word not in tmp:
                                list_of_abs_path.append(tmp)

            elif mode == 'file' or mode == 'all':
                if extension is None:
                    # non è stato definito alcun criterio di estrazione per estensione, quindi estraggo tutti i file
                    if filter_by_word is None:
                        # non filtro
                        list_of_abs_path.append(tmp)
                    else:
                        if not revert_filter:
                            if filter_by_word in tmp:
                                list_of_abs_path.append(tmp)
                        elif revert_filter:
                            if filter_by_word not in tmp:
                                list_of_abs_path.append(tmp)
                else:
                    # se è stata definita una estensione da cercare
                    if extension in tmp:
                        if filter_by_word is None:
                            # non filtro
                            list_of_abs_path.append(tmp)
                        else:
                            if not revert_filter:
                                if filter_by_word in tmp:
                                    list_of_abs_path.append(tmp)
                            elif revert_filter:
                                if filter_by_word not in tmp:
                                    list_of_abs_path.append(tmp)

        return list_of_abs_path
    def load_UCBM(self, path_to_dataset):
        '''
        Load the root directory of the UCBM internal dataset

        Parameters:
        - path_to_dataset: path to the dataset [str]

        Returns:
        - root_ucbm_content: the content of the root folder of the datasets [list[str]]
        '''

        root_ucbm_content = self.list_dir_with_relative_path_to(path_to_dataset)

        return root_ucbm_content

    def load_EndoSlam(self, path_to_dataset, mode = "testing"):
        '''
        Load the root directory of the EndoSlam dataset. If mode is "training" then it will load the training part of the
        dataset, otherwise if mode is "testing" it will load the testing part.

        Parameters:
        - path_to_dataset: path to the dataset [str]
        - mode: can be training/testing [str]

        Returns:
        - root_EndoSLam_content: the content of the root folder of the datasets [list[str]]
        '''

        endoslam_content = self.list_dir_with_relative_path_to(path_to_dataset)




        return endoslam_content

    def ucbm_dataloader(self, root_ucbm, batch_size, num_worker, i_folder):
        '''
        This function create the dataloader obj for the UCBM internal dataset

        Parameter:
        - root_ucbm: list of the root content [list[str]]
        - batch_size: the size of the batch to pass to the model during training
        - num_worker: the number of worker to use
        - i_folder: the ith folder to use in the list

        Return:
        - train_loader
        '''
        if i_folder >= len(root_ucbm):
            i_folder = 0

        # get a list of the content inside the ith folder
        rgb_folder = os.path.join(root_ucbm[i_folder], "rgb")
        depth_folder = os.path.join(root_ucbm[i_folder], "depth")
        training_frames = self.list_dir_with_relative_path_to(rgb_folder, mode = 'file', extension='.jpg', filter_by_word='dp', revert_filter=True)
        training_depth = self.list_dir_with_relative_path_to(depth_folder, mode='file', extension='.png')
        training_dataset = PoseDatasetLoader(training_frames, training_depth, dataset_type="UCBM")
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers = num_worker, pin_memory=True)

        return train_loader, i_folder


    def endoslam_dataloader(self, root_endoslam, batch_size, num_worker, i):
        '''
        This function create the dataloader obj for the EndoSlam dataset

        Parameter:
        - root_endoslam: list of the root content [list[str]]
        - batch_size: the size of the batch to pass to the model during training
        - num_worker: the number of worker to use
        - i_folder: the ith folder to use in the list

        Return:
        - test_loader
        '''
        rgb_folder = os.path.join(root_endoslam[i], "rgb")
        depth_folder = os.path.join(root_endoslam[i], "depth")
        testing_frames = self.list_dir_with_relative_path_to(rgb_folder, mode='file',
                                                        extension=".jpg", filter_by_word="dp", revert_filter=True)
        testing_depths = self.list_dir_with_relative_path_to(depth_folder, mode='file', extension='.png')

        testing_poses_path_xlsx = self.list_dir_with_relative_path_to(root_endoslam[i], mode="file", extension=".xlsx")
        print(testing_poses_path_xlsx)

        testing_poses = self.xlsxIO.read_xlsx_pose_file(testing_poses_path_xlsx[0])

        testing_dataset = PoseDatasetLoader(testing_frames, testing_depths, testing_poses, dataset_type="EndoSlam")
        test_loader = DataLoader(testing_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_worker,
                                 pin_memory=True)
        return test_loader