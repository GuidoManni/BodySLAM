'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Created on: 19/09/23

Description:
This file provide the metrics used for evaluating the pipeline
'''

# Computational Module
import numpy as np

# Metric lib
from evo.core import metrics
from evo.core import units
from evo.tools import file_interface

class MDEM_Metrics:
    '''
    This Class provide the metrics used for evaluating the Monocular Depth Estimation Module (MDEM)
    '''
    def __init__(self):
        pass
    def abs_rel_diff(self, prediction, ground_truth):
        '''
        This function computes the absolute relative difference between the depth map predicted (prediction) and the
        ground truth depth (ground_truth)
        :param prediction: depth map predicted
        :param ground_truth: ground truth depth
        :return: abs_rel
        '''

        mask = np.logical_and(ground_truth != 0, ~np.isnan(ground_truth))

        abs_rel = np.nanmean(np.abs(ground_truth[mask] - prediction[mask]) / (ground_truth[mask]))

        return abs_rel

    def squared_rel_error(self, prediction, ground_truth):
        '''
        This function computes the squared relative error between the depth map predicted (prediction) and the
        ground truth depth (ground_truth)
        :param prediction: depth map predicted
        :param ground_truth: ground truth depth
        :return: sq_rel
        '''

        mask = np.logical_and(ground_truth != 0, ~np.isnan(ground_truth))

        sq_rel = np.nanmean(((ground_truth[mask] - prediction[mask]) ** 2) / ground_truth[mask])

        return sq_rel

    def rmse(self, prediction, ground_truth):
        '''
        This function computes the root mean squared error between the depth map predicted (prediction) and the
        ground truth depth (ground_truth)
        :param prediction: depth map predicted
        :param ground_truth: ground truth depth
        :return: rmse
        '''
        mask = np.logical_and(ground_truth != 0, ~np.isnan(ground_truth))

        RMSE = (ground_truth[mask] - prediction[mask]) ** 2
        RMSE = np.sqrt(np.nanmean(RMSE))

        return RMSE

    def rmse_log(self, prediction, ground_truth):
        '''
        This function computes the log root mean squared error between the depth map predicted (prediction) and the
        ground truth depth (ground_truth)
        :param prediction: depth map predicted
        :param ground_truth: ground truth depth
        :return: rmse_log
        '''
        mask = np.logical_and(np.logical_and(ground_truth > 0, prediction > 0),
                              np.logical_and(~np.isnan(ground_truth), ~np.isnan(prediction)))

        # step 3: computiamo la metrica
        log_rmse = (np.log(ground_truth[mask]) - np.log(prediction[mask])) ** 2
        log_rmse = np.sqrt(np.nanmean(log_rmse))

        return log_rmse

    def accuracy_with_threshold(self, prediction, ground_truth, criterion):
        '''
        This function computes the accuracy between the depth map predicted (prediction) and the
        ground truth depth (ground_truth) with a fixed threshold

        :param prediction: depth map predicted
        :param ground_truth: ground truth depth
        :param criterion: criterion used for the computation of the threshold
        :return: accuracy
        '''
        mask = np.logical_and(ground_truth > 0, prediction > 0)

        treshold = np.maximum((ground_truth[mask] / prediction[mask]), (prediction[mask] / ground_truth[mask]))

        accuracy = (treshold < criterion ** 2).mean()

        return accuracy

class DepthMapMetric:
    def __init__(self, id, abs_rel_diff, squared_rel_error, rmse, rmse_log, accuracy_with_threshold):
        '''
        Class DepthMapMetric
        :param id:
        :param abs_rel_diff:
        :param squared_rel_error:
        :param rmse:
        :param rmse_log:
        :param accuracy_with_threshold:
        '''
        self.id = id
        self.abs_rel_diff = abs_rel_diff
        self.squared_rel_error = squared_rel_error
        self.rmse = rmse
        self.rmse_log = rmse_log
        self.accuracy_with_threshold = accuracy_with_threshold

class MPEM_Metrics:
    '''
    This class provides function used to evaluate the Monocular Pose Estimation Module (MPE)
    '''

    def __init__(self):
        # initialize metrics object
        self.ATE_metric = metrics.APE(metrics.PoseRelation.translation_part)
        self.RTE_metric = metrics.RPE(metrics.PoseRelation.translation_part)
        self.RRE_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_deg)

    def read_trajectory(self, path_to_trajectory):
        return file_interface.read_kitti_poses_file(path_to_trajectory)

    def compute_pose_metrics(self, path_to_gt_tr, path_to_pred_tr):
        # step 0: load the trajectory
        gt_tr = self.read_trajectory(path_to_gt_tr)
        pred_tr = self.read_trajectory(path_to_pred_tr)

        # align and correct the scale
        pred_tr.align_origin(gt_tr)
        pred_tr.align(gt_tr, correct_scale=True)

        # compute ATE, RTE & RRE
        data = (gt_tr, pred_tr)
        self.ATE_metric.process_data(data)
        self.RTE_metric.process_data(data)
        self.RRE_metric.process_data(data)

        # extract rmse and std
        ate_rmse = self.ATE_metric.get_statistic(metrics.StatisticsType.rmse)
        rte_rmse = self.RTE_metric.get_statistic(metrics.StatisticsType.rmse)
        rre_rmse = self.RRE_metric.get_statistic(metrics.StatisticsType.rmse)
        ate_std = self.ATE_metric.get_statistic(metrics.StatisticsType.std)
        rte_std = self.RTE_metric.get_statistic(metrics.StatisticsType.std)
        rre_std = self.RRE_metric.get_statistic(metrics.StatisticsType.std)

        results = {
            "ATE": (ate_rmse, ate_std),
            "RTE": (rte_rmse, rte_std),
            "RRE": (rre_rmse, rre_std),
        }

        return results






