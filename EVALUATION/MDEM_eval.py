'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Created on: 13/09/23

Description:
This script perform the evaluation on Zoe
'''
# Python standard lib
import os


# Internal Module
from MDEM.mdem_interface import MDEMInterface

# Other
from tqdm import tqdm



def read_Hamlyn(path_to_Hamlyn):
    '''
    This function obtain the full relative path of all the images.

    Directory Structure:
    Hamlyn
    -> rectified01
        -> image01
        -> image02
        -> depth01
        -> depth02
    ...
    -> rectified27

    Parameter:
    - path_to_Hamlyn: path to the dataset

    Return:
    - dict_of_path
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
                rectified_content_dict["depth01"] = list_content

            elif rectified_content[j] == "depth02":
                tmp = os.listdir(os.path.join(hamlyn_root_content_path[i], rectified_content[j]))
                for elem in tmp:
                    if '.png' in elem:
                        list_content.append(os.path.join(hamlyn_root_content_path[i], rectified_content[j], elem))
                rectified_content_dict["depth02"] = list_content

            elif rectified_content[j] == "image01":
                tmp = os.listdir(os.path.join(hamlyn_root_content_path[i], rectified_content[j]))
                for elem in tmp:
                    if '.jpg' in elem:
                        list_content.append(os.path.join(hamlyn_root_content_path[i], rectified_content[j], elem))
                rectified_content_dict["image01"] = list_content
            elif rectified_content[j] == "image02":
                tmp = os.listdir(os.path.join(hamlyn_root_content_path[i], rectified_content[j]))
                for elem in tmp:
                    if '.png' in elem:
                        list_content.append(os.path.join(hamlyn_root_content_path[i], rectified_content[j], elem))
                rectified_content_dict["image02"] = list_content

            dataset_paths[folder_name] = rectified_content_dict

    return dataset_paths



def read_EndoSlam(path_to_EndoSlam):
    # TODO: da completare
    pass





def compute_and_save_monocular_depth(dataset_type, dataset_paths, saving_path):
    '''
    This function compute the depth for each image in the evaluation dataset

    Parameter:
    - dataset_type: the dataset to use for validation (Hamlyn/EndoSlam)
    - dataset_paths: dict that contains the path of the dataset
    - saving_path: saving path for monocular depth map
    '''

    if dataset_type == "Hamlyn":
        img_key = "image01"
    elif dataset_type == "EmdoSlam":
        img_key = "" # TODO: inserisci la chiave appropriata

    for rectified_folder in dataset_paths.keys():
        print(f"[INFO]: computing depth for {rectified_folder}")
        for path in tqdm(dataset_paths[rectified_folder][img_key]):
            # compute the depth map
            dp = mdemint.infer_monocular_depth_map(path)

            # obtain the name of the depth map
            dp_name = path.split("/")[-1].replace(".jpg", ".png")

            # get the full path
            if not os.path.isdir(os.path.join(saving_path, rectified_folder)):
                os.mkdir(os.path.join(saving_path, rectified_folder))
            dp_path = os.path.join(saving_path, rectified_folder, dp_name)

            # save the dp
            mdemint.save_depth_map(dp, dp_path)








def evaluate_MDEM_on(dataset_type, path_to_dataset, saving_path, compute_depth = True):
    '''
    This function evaluate the MDEM on different datasets
    Parameter:
    - dataset_type: the dataset to use for validation (Hamlyn/EndoSlam)
    - path_eval_dataset: the path to the evaluation dataset [str]
    - saving_path: the saving path of the depth map [str]
    '''

    # step 0: we load the paths
    if dataset_type == "Hamlyn":
        dataset_paths = read_Hamlyn(path_to_dataset)
    elif dataset_type == "EndoSlam":
        dataset_paths = read_EndoSlam(path_to_dataset)

    # step 1: we compute the depth
    if compute_depth:
        compute_and_save_monocular_depth(dataset_type, dataset_paths, saving_path)







mdemint = MDEMInterface()

path_to_dataset = "/home/gvide/Dataset/Hamlyn_Dataset"
saving_path = "/home/gvide/Dataset/test/"
dataset_type = "Hamlyn"

evaluate_MDEM_on(dataset_type, path_to_dataset, saving_path)
#read_Hamlyn("/home/gvide/Dataset/Hamlyn_Dataset")

