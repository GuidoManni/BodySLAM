'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Created on: 20/09/23

Description:
This script perform the evaluation of the MPE module
'''
# Python standard modules
import os
import json
import csv

# Computational Module
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Metrics Module
import evo

# Other Modules
from tqdm import tqdm
import pandas as pd

# Internal Modules
from UTILS.io_utils import XlsxIO, CSVIO, TXTIO
from MPEM.mpem_interface import MPEMInterface
from UTILS.geometry_utils import PoseOperator
from evaluation_metrics import MPEM_Metrics

txtIO = TXTIO()

def convert_endoslam_to_kitti_format(input_path, output_path):
    """Convert the given .xlsx file to KITTI format and save to .txt file."""
    # Load the data

    poses = xlsxIO.read_xlsx_pose_file(input_path)

    txtIO.save_poses_as_kitti(poses, output_path)




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
        if os.path.isdir(os.path.join(path_to_dataset, content)):
            testing_root_paths.append(os.path.join(path_to_dataset, content))

    # step 2: now for each folder we extract the frames and xlsx pose file
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
                pose_path_xlsx = os.path.join(path, content_sub_folder[i])
                pose_path_kitti = pose_path_xlsx.replace(".xlsx", ".txt")
                convert_endoslam_to_kitti_format(pose_path_xlsx, pose_path_kitti)
        dict_path_content['Frames'] = sorted(list_of_frame_paths)
        dict_path_content['Poses'] = pose_path_kitti
        dataset_paths[curr_folder]=dict_path_content

    return dataset_paths


def read_scared_pose_format(path_to_json):
    '''
    This function read the json file which contains the pose information
    :param path_to_json:
    :return: the transformation matrix
    '''

    # load the calibration data from JSON
    with open(path_to_json, "r") as file:
        json_data = json.load(file)

    camera_pose = np.array(json_data["camera-pose"])

    return camera_pose


def extract_data_from_matrices(motion_matrices):
    extracted_data = []

    for motion_matrix in motion_matrices:
        # Extract translation vector
        t = motion_matrix[:3, 3]

        # Extract rotation matrix
        R = motion_matrix[:3, :3]

        # Convert rotation matrix to quaternion
        quat = Rotation.from_matrix(R).as_quat()

        # Append extracted data to the list
        extracted_data.append(np.concatenate([t, quat]))

    return extracted_data


def save_poses_to_csv(motion_matrices, filename):
    # Extract data from motion matrices
    extracted_data = extract_data_from_matrices(motion_matrices)

    # Write the extracted data to CSV
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        csv_writer.writerow(["trans_x", "trans_y", "trans_z", "quat_x", "quat_y", "quat_z", "quat_w"])

        # Write extracted data
        for row in extracted_data:
            csv_writer.writerow(row)


def load_SCARED_poses(path_to_dataset):
    '''
    This function load the content of the SCARED Dataset
    :param path_to_dataset:
    :return: a dict of poses
    '''

    '''
    This function works only with the following folder architecture
    [SCARED]
    -> [dataset_1_kf_1]
        -> [frame_data]
            -> frame_data00000.json
            -> ...
        -> ...
    -> ...
    '''

    # step 0: get the content of the root folder
    root_content = os.listdir(path_to_dataset)

    # step 1: get the full relative path
    poses_paths = []
    frame_paths = []
    for content in root_content:
        poses_paths.append(os.path.join(path_to_dataset, content, "frame_data"))
        frame_paths.append(os.path.join(path_to_dataset, content, 'left'))


    # step 2: now for each dataset we will create a list to store the loaded poses
    dataset_dict = {}
    for path in frame_paths:
        dataset = path.split("/")[-2]
        frames_path = os.listdir(path)
        poses_path = os.listdir(path.replace('left', "frame_data"))
        poses = []
        for pose_path in poses_path:
            camera_pose = read_scared_pose_format(pose_path)
            poses.append(camera_pose)

        dict = {
            'Poses': poses,
            'Frames': frames_path
        }
        dataset_dict[dataset] = dict

    return dataset_dict


def save_poses_as_kitti(poses_list, output_path):
    """Save a list of 4x4 numpy array poses in KITTI format after ensuring SO(3) validity."""

    # Ensure that all rotation matrices are in SO(3)
    corrected_poses = []
    for pose in poses_list:
        corrected_pose = np.copy(pose)
        #corrected_pose[:3, :3] = poseOperator.ensure_so3(pose[:3, :3])
        corrected_poses.append(corrected_pose)

    # Save the poses to a .txt file with each pose on one line
    with open(output_path, 'w') as f:
        for pose in corrected_poses:
            # Flatten the pose matrix and write as a single line
            f.write(" ".join(map(str, pose.flatten()[:-4])) + "\n")


def compute_poses(dataset_paths, saving_path_for_prediction):
    '''
    This function compute the poses for a given dataset using the cyclepose
    :param dataset_paths: a dict that contains the whole directory path of the dataset being passed
    :param saving_path_for_prediction: the save path where to store the results
    :return
    '''

    poses_path = {}
    # to compute the poses we need to infer it using the cyclepose
    for folder in dataset_paths.keys():
        poses_list = []
        num_of_frames = len(dataset_paths[folder]['Frames'])
        initial_pose = np.eye(4)
        absolute_pose = initial_pose
        absolute_poses = [initial_pose]
        for i in tqdm(range(num_of_frames)):
            if i < num_of_frames - 1:
                path_to_frame1 = dataset_paths[folder]['Frames'][i]
                path_to_frame2 = dataset_paths[folder]['Frames'][i+1]
                relative_pose = mpem_interface.infer_relative_pose_between(path_to_frame1, path_to_frame2)
                absolute_pose = absolute_pose @ relative_pose
                absolute_pose[:3, :3] = poseOperator.ensure_so3_v2(absolute_pose[:3, :3])
                absolute_poses.append(absolute_pose)

        file_name = folder + ".txt"
        save_poses_as_kitti(absolute_poses, os.path.join(saving_path_for_prediction, file_name))

        poses_path[folder] = os.path.join(saving_path_for_prediction, file_name)

    return poses_path



def save_results(folder, results_path, metrics):
    '''
    This function save in a .csv file the results computed
    :param folder:
    :param results_path:
    :return:
    '''
    saving_folder = os.path.join(results_path, folder)
    if not os.path.exists(saving_folder):
        os.mkdir(saving_folder)
    saving_path = os.path.join(results_path, folder, "results.csv")
    csvIO.write_metrics_on_cvs(saving_path, metrics)




def compute_metrics(dataset_paths, predictions_path, results_path):
    for folder in predictions_path.keys():
        print(f"[INFO]: computing metrics for {folder}")

        # step 0: get the gt path
        gt_traj_path = dataset_paths[folder]["Poses"]
        # step 1: get the pred path
        pred_traj_path = predictions_path[folder]

        print(gt_traj_path)
        print(pred_traj_path)

        # step 2: compute the metrics
        results = mpem_metrics.compute_pose_metrics(gt_traj_path, pred_traj_path)

        # step 3: save the results
        path_to_metric = os.path.join(results_path, folder)
        file_name = path_to_metric + ".csv"

        with open(file_name, 'w', newline='') as csvfile:
            fieldnames = ['Metric', 'Value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for key, value in results.items():
                writer.writerow({'Metric': key, 'Value': value})



    



def evaluate_MPEM(dataset_type, path_to_dataset, saving_path_for_prediction, results_path, compute_pose = True):
    '''
    This function compute the metrics for the evaluation of the MPE module on the EndoSlam and SCARED dataset
    :param dataset_type: the type of dataset to be used
    :param path_to_dataset: path to the dataset
    :param saving_path_for_prediction: the save path for the poses
    :param results_path: the save path where to store the results
    :param compute_pose: True if we want to compute the pose
    :return:
    '''

    # step 1: we load the testing dataset
    if dataset_type == "EndoSlam":
        dataset_paths = load_EndoSlam_testing_pose(path_to_dataset)
    elif dataset_type == "SCARED":
        dataset_paths = load_SCARED_poses(path_to_dataset)

    print(dataset_paths)

    # step 2: if requested we compute the predicted poses
    if compute_pose:
        predictions_path = compute_poses(dataset_paths, saving_path_for_prediction)
    else:
        prediction_path_tmp = os.listdir(saving_path_for_prediction)
        print(prediction_path_tmp)
        predictions_path = {}
        for elem in prediction_path_tmp:
            predictions_path[elem.replace(".txt", "")] = os.path.join(saving_path_for_prediction, elem)


    # step 3: we compute ARE, ATE and RRE and RTE
    compute_metrics(dataset_paths, predictions_path, results_path)


poseOperator = PoseOperator()
xlsxIO = XlsxIO()
csvIO = CSVIO()
mpem_metrics = MPEM_Metrics()

path_to_dataset = "/home/gvide/Dataset/EndoSlam_testing"
saving_path_for_prediction = "/home/gvide/Scrivania/test"
results_path = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/BodySLAM/EndoSLAM_Results/"
#path_to_model = "/home/gvide/PycharmProjects/SurgicalSlam/MPEM/Model/15_best_model_PaD_B.pth"
path_to_model = "/home/gvide/PycharmProjects/SurgicalSlam/MPEM/Model/15_PaD_B.pth"
mpem_interface = MPEMInterface(path_to_model=path_to_model)
evaluate_MPEM("EndoSlam", path_to_dataset, saving_path_for_prediction, results_path, True)