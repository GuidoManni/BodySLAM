'''
Compute the metrics for competitor
'''
import os
import sys
from evaluation_metrics import MPEM_Metrics
import csv
pose_metrics = MPEM_Metrics()
import numpy as np


def ensure_so3_v2(matrix):
    """
    Projects a 3x3 matrix to the closest SO(3) matrix using an alternative method.
    """
    try:
        U, _, Vt = np.linalg.svd(matrix)
    except:
        print(matrix)
        sys.exit()

    # Creating the intermediate diagonal matrix
    D = np.eye(3)
    D[2, 2] = np.linalg.det(U) * np.linalg.det(Vt)

    # Compute the closest rotation matrix
    R = np.dot(U, np.dot(D, Vt))

    return R

def correct_poses(input_file_path, output_file_path):
    """
    Reads poses from the input file, corrects the rotation matrices, and saves the corrected poses to the output file.

    :param input_file_path: Path to the file containing the original poses.
    :param output_file_path: Path where the corrected poses will be saved.
    """

    corrected_poses = []

    with open(input_file_path, 'r') as file:
        for line in file:
            pose = np.array(line.strip().split(), dtype=float)
            pose_matrix = pose.reshape((3, 4))

            rotation_matrix = pose_matrix[:, :3]
            translation_vector = pose_matrix[:, 3].reshape(3, 1)

            corrected_rotation_matrix = ensure_so3_v2(rotation_matrix)

            corrected_pose = np.hstack((corrected_rotation_matrix, translation_vector))
            corrected_poses.append(corrected_pose)

    corrected_poses_array = np.array(corrected_poses)

    # Saving the corrected poses to the output file
    with open(output_file_path, 'w') as file:
        for pose in corrected_poses_array:
            pose_string = ' '.join(['{:.8e}'.format(num) for num in pose.flatten()])
            file.write(pose_string + '\n')

    print(f"Corrected poses saved to {output_file_path}")



def load_EndoSLAM_gt(root_path_gt: str) -> list:
    root_content = os.listdir(root_path_gt)
    path_to_gts = []
    
    # reconstruct the full path
    for folder in root_content:
        if ".py" not in folder:
            if os.path.isdir(os.path.join(root_path_gt, folder)):
                folder_content = os.listdir(os.path.join(root_path_gt, folder))
            for file in folder_content:
                if ".txt" in file and "list_of_frame" not in file:
                    path_to_txt = os.path.join(root_path_gt, folder) + "/" + file
                    path_to_gts.append(path_to_txt)

    return sorted(path_to_gts)

def load_pds(root_path_pd: str) -> list:
    root_content = os.listdir(root_path_pd)

    path_to_pds = []

    # reconstruct the full path
    for file in root_content:
        if os.path.isfile(os.path.join(root_path_pd, file)):
            if ".txt" in file:
                path_to_txt = os.path.join(root_path_pd, file)
                path_to_pds.append(path_to_txt)

    return sorted(path_to_pds)

        
        
def load_SCARED_gt(root_path_gt: str) -> list:
    root_content = os.listdir(root_path_gt)

    path_to_gts = []

    # reconstruct the full path
    for file in root_content:
        if os.path.isfile(os.path.join(root_path_gt, file)):
            if ".txt" in file:
                path_to_txt = os.path.join(root_path_gt, file)
                path_to_gts.append(path_to_txt)

    return sorted(path_to_gts)
    

def compute_pose_metrics(root_path_gt: str, root_path_pd: str, dataset_type: str):
    if dataset_type == "EndoSLAM":
        path_to_gts = load_EndoSLAM_gt(root_path_gt)
    elif dataset_type == "SCARED":
        path_to_gts = load_SCARED_gt(root_path_gt)
    else:
        print(f"[INFO]: The dataset '{dataset_type}' selected does not exist! Try 'EndoSLAM' or 'SCARED'")
        sys.exit()

    path_to_pds = load_pds(root_path_pd)
    #print(path_to_gts)
    #print(path_to_pds)

    for path in path_to_pds:
        correct_poses(path, path)


    if len(path_to_gts) == len(path_to_pds):
        for i in range(len(path_to_gts)):
            print(f"[INFO]: computing poses between {path_to_gts[i]} - {path_to_pds[i]}")
            tmp = path_to_pds[i].split("/")[1:8]
            file_name = path_to_pds[i].split("/")[-1].replace(".txt", "")
            saving_path = ""
            for path in tmp:
                saving_path = os.path.join(saving_path, path)
            saving_path += "/Final_Results/"
            saving_path = saving_path.replace("home/", "/home/")



            results = pose_metrics.compute_pose_metrics(path_to_gts[i], path_to_pds[i])

            file_name = saving_path + file_name + ".csv"

            with open(file_name, 'w', newline='') as csvfile:
                fieldnames = ['Metric', 'Value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for key, value in results.items():
                    writer.writerow({'Metric': key, 'Value': value})


            #print(path_to_results)
    else:
        for i in range(len(path_to_gts)):
            print(f"{path_to_gts[i]} - {path_to_pds[i]}")
        raise ValueError(
            f"Mismatch between the number of loaded gt and the number of loaded pd ({len(path_to_gts)}/{len(path_to_pds)})")






# EndoSLAM
root_path_gt = "/home/gvide/Dataset/EndoSlam_testing"
root_path_pd = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/EndoDepth/EndoSLAM/Results"
dataset_type = "EndoSLAM"

# SCARED
#root_path_gt = "/home/gvide/Dataset/SCARED_Pose_GT"
#root_path_pd = "/home/gvide/Scrivania/BodySLAM Results/MPEM Validation/EndoDepth/SCARED/Results"
#dataset_type = "SCARED"

compute_pose_metrics(root_path_gt, root_path_pd, dataset_type)