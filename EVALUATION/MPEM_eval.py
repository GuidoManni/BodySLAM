'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Created on: 20/09/23

Description:
This script perform the evaluation of the MPE module
'''
# Python standard modules
import os


# Other Modules
from tqdm import tqdm

# Internal Modules
from UTILS.io_utils import XlsxIO
from MPEM.mpem_interface import MPEMInterface
from UTILS.geometry_utils import PoseOperator
from evaluation_metrics import MPEM_Metrics



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
                pose_path = os.path.join(path, content_sub_folder[i])
        dict_path_content['Frames'] = sorted(list_of_frame_paths)
        dict_path_content['Poses'] = pose_path
        dataset_paths[curr_folder]=dict_path_content

    return dataset_paths



def compute_poses(dataset_paths, saving_path_for_prediction):
    '''
    This function compute the poses for a given dataset using the cyclepose
    :param dataset_paths: a dict that contains the whole directory path of the dataset being passed
    :param saving_path_for_prediction: the save path where to store the results
    :return
    '''

    poses_dict = {}
    # to compute the poses we need to infer it using the cyclepose
    for folder in dataset_paths.keys():
        poses_list = []
        num_of_frames = len(dataset_paths[folder]['Frames'])
        for i in tqdm(range(num_of_frames)):
            if i > 0:
                path_to_frame1 = dataset_paths[folder]['Frames'][i-1]
                path_to_frame2 = dataset_paths[folder]['Frames'][i]
                relative_pose = mpem_interface.infer_relative_pose_between(path_to_frame1, path_to_frame2)
                poses_list.append(relative_pose)
        poses_dict[folder] = poses_list

    print(poses_dict)
    # TODO: implementa funzione di salvataggio
    return poses_dict




def compute_metrics(dataset_paths, predictions, results_path):
    '''
    This function compute the ATE, ARE, RRE and RTE
    :param dataset_paths: a dict that contains the ground truth absolute poses
    :param predictions: a dict that contains the relative prediciton poses
    :param results_path: where to save the results
    :return: 
    '''
    METRICS = {}

    for folder in dataset_paths.keys():
        # step 0: we load the xlsx file containing the absolute ground truth poses
        ground_truth = xlsxIO.read_xlsx_pose_file(dataset_paths[folder]['Poses'])

        # step 1: compute the abs and rel poses respectively for the predicitons and the ground_truth
        abs_prediction = poseOperator.integrate_relative_poses(predictions[folder])
        rel_ground_truth = poseOperator.get_relative_poses(ground_truth)

        # step 2: compute the metrics
        ATE, ARE = mpem_metrics.absolute_pose_error(ground_truth, abs_prediction)
        RRE, RTE = mpem_metrics.compute_RRE_and_RTE(rel_ground_truth, predictions[folder])
        print(f"FOLDER {folder}")
        print(f"ATE: {ATE}")
        print(f"ARE: {ARE}")
        print(f"RRE: {RRE}")
        print(f"RTE: {RTE}")

    



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
        predictions = compute_poses(dataset_paths, saving_path_for_prediction)

    # step 3: we compute ARE, ATE and RRE and RTE
    compute_metrics(dataset_paths, predictions, results_path)


poseOperator = PoseOperator()
xlsxIO = XlsxIO()
mpem_metrics = MPEM_Metrics()

path_to_dataset = "/home/gvide/Dataset/EndoSlam_testing"
saving_path_for_prediction = ""
results_path = ""
path_to_model = "/home/gvide/PycharmProjects/SurgicalSlam/MPEM/Model/5_best_model_PaD_A.pth"
mpem_interface = MPEMInterface(path_to_model=path_to_model)
evaluate_MPEM(path_to_dataset, saving_path_for_prediction, results_path)