'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Created on: 20/09/23

Description:
This script perform the evaluation of the MPE module
'''
# Python standard modules
import os


# Internal Modules
from UTILS.io_utils import DatasetLoader


def compute_poses(dataset_paths, saving_path_for_prediction, method):
    '''
    This function compute the poses using a selected method
    :param dataset_paths:
    :param saving_path_for_prediction:
    :param method: can be icp_pt2pt, icp_pt2pl, pht_st, pht_park, pht_endo_dp, cyclepose, all
    :return:
    '''

    if method == 'icp_pt2pt':
        pass
    elif method == 'icp_pt2pl':
        pass
    elif method == 'pht_st':
        pass
    elif method == 'pht_park':
        pass
    elif method == 'pht_endo_dp':
        pass
    elif method == 'cyclepose':
        pass
    elif method == 'all':
        pass

def load_EndoSlam_testing_pose(path_to_dataset):
    '''
    This function load the content of the testing set
    :param path_to_dataset: path to the EndoSlam testing set
    :return: dataset_paths: dict containing full relative path of the content of the folder
    '''

    '''
    This function works only with the following architecture of the testing folder
    [Testing Folder]
    -> [Folder1]
        -> frame000000.jpg
        -> ...
        -> pose.xlsx
    ...
    -> [FolderN]
        -> frame000000.jpg
        -> ...
        -> pose.xlsx
    '''

    dataset_paths = {}

    # step 1: get the content of the root folder
    testing_root_content = os.listdir(path_to_dataset)

    # we build the full relative path
    testing_root_paths = []
    for content in testing_root_content:
        testing_root_paths.append(os.path.join(path_to_dataset, content))

    # step 2: now for each folder we exctract the frames and xlsx pose file
    for path in testing_root_paths:
        curr_folder = path.split("/")[-1]
        dict_path_content = {}

        # get the content
        print(f"Loading content from {curr_folder}")
        content_sub_folder = os.listdir(path)

        list_of_frame_paths = []
        for i in range(len(content_sub_folder)):
            if ".jpg" in content_sub_folder[i]:
                list_of_frame_paths.append(os.path.join(path, content_sub_folder[i]))
            elif '.xlsx' in content_sub_folder[i]:
                pose_path = content_sub_folder[i]
        dict_path_content['Frames'] = list_of_frame_paths
        dict_path_content['Poses'] = pose_path
        dataset_paths[curr_folder]=dict_path_content

    return dataset_paths



def compute_pose(dataset_paths, saving_path_for_prediction):
    '''
    This function compute the pposes for a given dataset using the cyclepose
    :param dataset_paths: a dict that contains the whole directory path of the dataset being passed
    :saving_path_for_prediction: the save path where to store the results
    '''
    



def evaluate_MPEM(path_to_dataset, saving_path_for_prediction, results_path, compute_pose = True):
    '''
    This function compute the metrics for the evaluation of the MPE module on the EndoSlam dataset
    :param path_to_dataset: path to the dataset
    :param saving_path_for_prediction: the save path for the poses
    :param results_path: the save path where to store the results
    :param compute_pose: True if we want to compute the pose
    :return:
    '''

    # step 1: we load the testing dataset
    dataset_paths = load_EndoSlam_testing_pose(path_to_dataset)

    print(dataset_paths)

    # step 2: if requested we compute the predicted poses
    if compute_pose:
        compute_poses(dataset_paths, saving_path_for_prediction)

path_to_dataset = "/home/gvide/Dataset/EndoSlam_testing"
saving_path_for_prediction = ""
results_path = ""
evaluate_MPEM(path_to_dataset, saving_path_for_prediction, results_path)