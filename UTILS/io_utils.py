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

# Internal module
from MPEM.dataloader import PoseDatasetLoader
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



class DatasetsIO:

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

        for content in endoslam_content:
            if mode in content:
                root_EndoSLam_content = self.list_dir_with_relative_path_to(content)

        return root_EndoSLam_content

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
        training_frames = self.list_dir_with_relative_path_to(root_ucbm[i_folder], mode = 'file', extension='.jpg', filter_by_word='dp', revert_filter=True)

        training_dataset = PoseDatasetLoader(training_frames, dataset_type="UCBM")
        train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_worker = num_worker, pin_memory=True)

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

        testing_frames = self.list_dir_with_relative_path_to(root_endoslam[i], mode='file',
                                                        extension=".jpg", filter_by_word="dp", revert_filter=True)
        testing_poses_path_xlsx = self.list_dir_with_relative_path_to(root_endoslam[i], mode="file", extension=".xlsx")

        testing_poses = XlsxIO.read_xlsx_pose_file(testing_poses_path_xlsx[0])

        testing_dataset = PoseDatasetLoader(testing_frames, testing_poses)
        test_loader = DataLoader(testing_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_worker,
                                 pin_memory=True)
        return test_loader


class XlsxIO:
    def read_xlsx_pose_file(self, filepath):
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

            # Convert to lie space for training purposes
            pose_se3 = LieEuclideanMapper.SE3_to_se3(transformation_matrix)

            motion_matrices.append(pose_se3)

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
                        'best_loss': checkpoint['loss']}
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
            saving_path = saving_path.replace("model", "best_model")
        torch.save({
            'epoch': training_var['epoch'],
            'iter_on_ucbm': training_var['iter_on_ucbm'],
            'loss': training_var['loss'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, saving_path)





