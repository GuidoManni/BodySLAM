'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Created on: 13/09/23

Description:
This script perform the evaluation on Zoe
'''
# Python standard lib
import os

import numpy as np

# Internal Module
from MDEM.mdem_interface import MDEMInterface
from evaluation_metrics import MDEM_Metrics, DepthMapMetric
from UTILS.io_utils import FrameIO, CSVIO, DatasetLoader
from UTILS.image_processing_utils import ImageProc

# Other
from tqdm import tqdm
import pandas as pd


def read_zoe_prediction(path_to_prediction):
    '''
    This function obtain the full relative path of all the predictions
    :param path_to_prediction:
    :return: dict_of_path
    '''

    '''
    This script for the Hamlyn dataset generate the following folder architecture:
    
    Prediction Folder
    -> rectified01
        -> 0000000001.png
        -> 0000000002.png
        -> ...
    -> rectified02
    -> ...
    '''

    prediction_paths = {}

    # step 1: we read the content of the root folder
    prediction_root_content = os.listdir(path_to_prediction)
    print("[INFO]: found the following content in the folder provided")
    print(prediction_root_content)

    # step 2: we remove in the list any eventual folder that is not "rectified"
    prediction_root_content_path = []
    for i in range(len(prediction_root_content)):
        if "rectified" in prediction_root_content[i] or "Colon" in prediction_root_content[i] or "Small Intestine" in prediction_root_content[i] or "Stomach" in prediction_root_content[i] or "dataset" in prediction_root_content[i]:
            prediction_root_content_path.append(os.path.join(path_to_prediction, prediction_root_content[i]))
    prediction_root_content_path = sorted(prediction_root_content_path)

    # step 3: for each rectified folder we get the content and save it in a dict (where the key is the rectified folder)
    for i in range(len(prediction_root_content_path)):
        # get the key (the rectified folder)
        key = prediction_root_content_path[i].split("/")[-1]

        # read the content of the current rectified folder
        predictions = sorted(os.listdir(prediction_root_content_path[i]))

        # we construct the full relative path
        full_relative_path = []
        for prediction in predictions:
            complete_path = os.path.join(prediction_root_content_path[i], prediction)
            full_relative_path.append(complete_path)

        prediction_paths[key] = full_relative_path
    #print(prediction_paths)

    return prediction_paths


def compute_and_save_monocular_depth(dataset_type, dataset_paths, saving_path):
    '''
    his function compute the depth for each image in the evaluation dataset

    :param dataset_type: the dataset to use for validation (Hamlyn/EndoSlam)
    :param dataset_paths: dict that contains the path of the dataset
    :param saving_path: saving path for monocular depth map
    :return:
    '''


    if dataset_type == "Hamlyn":
        img_key = "image01"
    elif dataset_type == "EndoSlam":
        img_key = "Frames"
    elif dataset_type == "SCARED":
        img_key = "left"

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


def compute_median_scale_factor(ground_truth, predictions):
    """
    Compute the median scale factor between ground truth and predictions.

    Args:
    - ground_truth (numpy array): The ground truth depth maps.
    - predictions (numpy array): The predicted depth maps.

    Returns:
    - float: The median scale factor.
    """
    # Compute the scale factor using the median ratio
    s = np.median(ground_truth) / np.median(predictions)
    return s


def compute_metrics_for(dataset_type, dataset_paths, path_to_prediction, results_path):
    '''
    This function compute the actual metrics used for validating ZOE
    :param dataset_type: the dataset being used (Hamlyn/EndoSlam)
    :param dataset_paths: dict containing the path of the files in the dataset folder
    :param path_to_prediction: the path to the prediction
    :param results_path: path where results are stored
    :return:
    '''

    if dataset_type == "Hamlyn":
        depth_folder = "depth01"
    elif dataset_type == "EndoSlam":
        depth_folder = "Pixelwise Depths"
    elif dataset_type == "SCARED":
        depth_folder = "left_dp"

    # step 1: we get a dict of predictions
    #print(path_to_prediction)
    predictions_path = read_zoe_prediction(path_to_prediction)
    #print(predictions_path)

    # step 2: we compute the metrics



    print("\n[INFO]: Computing depth metrics...")
    for rectified_folder in predictions_path.keys():
        METRICS = []
        print(f"\n[INFO]: computing depth metrics for {rectified_folder}")
        avg_abs_rel_diff = 0.0
        avg_squared_rel_err = 0.0
        avg_rmse = 0.0
        avg_rmse_log = 0.0
        avg_accuracy = [0.0, 0.0, 0.0]

        total_depths = len(predictions_path[rectified_folder])
        for i in tqdm(range(total_depths)):
            # get the path
            prediction_path = predictions_path[rectified_folder][i]
            ground_truth_path = dataset_paths[rectified_folder][depth_folder][i]
            #print(prediction_path)
            #print(ground_truth_path)

            # load the depth maps
            prediction = frameIO.load_cv2_depth(prediction_path)
            ground_truth = frameIO.load_cv2_depth(ground_truth_path)


            # To compare the results with Endo-Depth-And-Motion we need to clip the ground truth
            if dataset_type == "Hamlyn":
                # We now apply some preprocessing
                # For the Hamlyn the saturation is between 1.0 mm to 300 mm, for this reason we are going to ignore the values that are not in this range
                valid_mask = np.logical_and(ground_truth > 1.0, ground_truth < 300)
                ground_truth = ground_truth[valid_mask]
                prediction = prediction[valid_mask]


            elif dataset_type == "SCARED":
                # since the SCARED dataset provide sparse depth map we need to create a mask containing only non zero valuse
                print(np.amax(ground_truth))
                non_zero_px = ground_truth > 0
                ground_truth = ground_truth[non_zero_px]
                prediction = prediction[non_zero_px]


            # we compute a median scale factor due to the fact that we are working with monocular images (Scale Ambiguity)
            s = compute_median_scale_factor(ground_truth, prediction)
            prediction = s * prediction


            # now we compute the metrics for each pair
            #print("\n")
            #print(ground_truth_path.split("/")[-1], prediction_path.split("/")[-1])
            id_ground_truth = ground_truth_path.split("/")[-1].split("_")[-1].replace(".png", "")
            id_prediction = prediction_path.split("/")[-1].split("_")[-1].replace(".png", "")
            if dataset_type == "SCARED":
                id_ground_truth = id_ground_truth.replace("points", "")
                id_prediction = id_prediction.replace("data", "")
            #assert id_ground_truth == id_prediction, "The pair ground_truth - prediction don't match!"
            abs_rel_diff = mdem_metrics.abs_rel_diff(prediction, ground_truth)
            squared_rel_err = mdem_metrics.squared_rel_error(prediction, ground_truth)
            rmse = mdem_metrics.rmse(prediction, ground_truth)
            rmse_log = mdem_metrics.rmse_log(prediction, ground_truth)
            accuracy_with_threshold_1 = mdem_metrics.accuracy_with_threshold(prediction, ground_truth, criterion=1.25)
            accuracy_with_threshold_2 = mdem_metrics.accuracy_with_threshold(prediction, ground_truth, criterion=1.25**2)
            accuracy_with_threshold_3 = mdem_metrics.accuracy_with_threshold(prediction, ground_truth, criterion=1.25**3)

            avg_abs_rel_diff += abs_rel_diff
            avg_squared_rel_err += squared_rel_err
            avg_rmse += rmse
            avg_rmse_log += rmse_log
            avg_accuracy[0] += accuracy_with_threshold_1
            avg_accuracy[1] += accuracy_with_threshold_2
            avg_accuracy[2] += accuracy_with_threshold_3

            metric = {
                'abs_rel_diff': abs_rel_diff,
                'squared_rel_err': squared_rel_err,
                'rmse': rmse,
                'rmse_log': rmse_log,
                'accuracy_1.25': accuracy_with_threshold_1,
                'accuracy_(1.25)^2': accuracy_with_threshold_2,
                'accuracy_(1.25)^3': accuracy_with_threshold_3
            }
            METRICS.append(metric)

        if not os.path.isdir(results_path):
            os.mkdir(results_path)

        results_rectified_path = os.path.join(results_path, rectified_folder)
        if not os.path.isdir(results_rectified_path):
            os.mkdir(results_rectified_path)
        results_name = os.path.join(results_rectified_path, "results.csv")

        csvIO.write_metrics_on_cvs(results_name, METRICS)

        # now we load it with pandas
        data = pd.read_csv(results_name)

        # compute the mean
        data_mean = data.mean()

        # save it to a csv file
        avg_results_name = os.path.join(results_rectified_path, "avg.csv")
        data_mean.to_csv(avg_results_name, header=True)






def evaluate_MDEM_on(dataset_type, path_to_dataset, saving_path_for_prediction, results_path, compute_depth = True):
    '''
    This function evaluate the MDEM on different datasets
    :param dataset_type: the dataset to use for validation (Hamlyn/EndoSlam/SCARED)
    :param path_to_ground_truth: the path to the ground_truth depth map
    :param path_to_rgb: the path to the rgb images
    :param saving_path_for_prediction: the path where to save the predictions
    :param compute_depth: if True -> compute depth map
    :return:
    '''

    assert dataset_type == "Hamlyn" or dataset_type == "EndoSlam" or dataset_type == "SCARED", "[ERROR]: DATASET TYPE MUST BE 'Hamlyn' or 'EndoSlam'"
    # step 0: we load the paths
    if dataset_type == "Hamlyn":
        dataset_paths = dataset_loader.read_Hamlyn(path_to_dataset)
    elif dataset_type == "EndoSlam":
        dataset_paths = dataset_loader.read_EndoSlam(path_to_dataset)
    elif dataset_type == "SCARED":
        dataset_paths = dataset_loader.read_SCARED(path_to_dataset)

    print(dataset_paths.keys())

    # step 1: we compute the depth
    if compute_depth:
        compute_and_save_monocular_depth(dataset_type, dataset_paths, saving_path)

    # step 2: we evaluate the results with the GT

    compute_metrics_for(dataset_type, dataset_paths, saving_path_for_prediction, results_path)






mdemint = MDEMInterface()
mdem_metrics = MDEM_Metrics()
frameIO = FrameIO()
csvIO = CSVIO()
imgeproc = ImageProc()
dataset_loader = DatasetLoader()

# Dataset Type
#dataset_type = "Hamlyn"
#dataset_type = "EndoSlam"
dataset_type = "SCARED"


# Dataset Path
#path_to_dataset = "/home/gvide/Dataset/Hamlyn_Dataset/"
#path_to_dataset = "/home/gvide/Dataset/EndoSlam/"
path_to_dataset = "/home/gvide/Dataset/SCARED/"

# Path to depth prediction for Hamlyn
#saving_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/BodySLAM/Depth_Map_Generated_with_Zoe/Hamlyn_zoe/"
#saving_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/EndoSfMLearner/depth_map_generated_by_EndoSFMLearner/EndoSfMLearner_results_Hamlyn/"

# Path to depth prediction for EndoSlam
#saving_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/BodySLAM/Depth_Map_Generated_with_Zoe/EndoSlam_zoe/"
#saving_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/EndoSfMLearner/depth_map_generated_by_EndoSFMLearner/EndoSfMLearner_results_EndoSlam/"
#saving_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/Endo_Depth/Depth_map_generated_by_Endo_Depth/dp_endo_depth_on_endoslam/"

# Path to depth prediction for SCARED
saving_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/BodySLAM/Depth_Map_Generated_with_Zoe/SCARED_zoe/"
#saving_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/Endo_Depth/Depth_map_generated_by_Endo_Depth/dp_endo_depth_on_SCARED"
#saving_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/EndoSfMLearner/depth_map_generated_by_EndoSFMLearner/EndoSfMLearner_results_SCARED/"


# Path to results for Hamlyn
#results_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/BodySLAM/Hamlyn_results/"
#results_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/EndoSfMLearner/Hamlyn_results/"

# Path to results for EndoSlam
#results_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/BodySLAM/EndoSLAM_results/"
#results_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/EndoSfMLearner/EndoSLAM_results/"
#results_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/Endo_Depth/results_endoslam_endo_depth"

# Path to results for SCARED
results_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/BodySLAM/SCARED_results/"
#results_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/Endo_Depth/results_scared_endo_depth/"
#results_path = "/home/gvide/Scrivania/BodySLAM Results/MDEM Validation/EndoSfMLearner/SCARED_results/"



evaluate_MDEM_on(dataset_type, path_to_dataset, saving_path, results_path, False)
#read_Hamlyn("/home/gvide/Dataset/Hamlyn_Dataset")

