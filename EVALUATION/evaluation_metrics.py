'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Created on: 19/09/23

Description:
This file provide the metrics used for evaluating the pipeline
'''

# Computational Module
import numpy as np

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
    This class provides function used to evaluatethe Monocular Pose Estimation Module (MPE)
    '''

    def __init__(self):
        pass

    def absolute_pose_error(self, ground_truth, predictions):
        """
        Computes the ATE and ARE.

        Parameters:
        ground_truth: List of numpy arrays representing the ground truth poses.
        predictions: List of numpy arrays representing the predicted poses.

        Returns:
        ate: Absolute Trajectory Error
        are: Absolute Rotation Error
        """
        assert len(ground_truth) == len(predictions), "Ground truth and predictions must have the same length"

        ate_sum = 0.0
        are_sum = 0.0

        for gt, pred in zip(ground_truth, predictions):
            # Ensure the pose matrices are 4x4 (assuming they are homogeneous matrices)
            assert gt.shape == (4, 4) and pred.shape == (4, 4), "Poses should be 4x4 matrices"

            # Compute translational error
            trans_diff = gt[:3, 3] - pred[:3, 3]
            ate_sum += np.linalg.norm(trans_diff)

            # Compute rotational error
            R_diff = np.dot(gt[:3, :3], pred[:3, :3].T)
            trace = np.trace(R_diff)
            angular_error_rad = np.arccos(max(min((trace - 1) / 2, 1), -1))
            are_sum += angular_error_rad

        ATE = ate_sum / len(ground_truth)
        ARE = are_sum / len(ground_truth)

        return ATE, ARE

    def compute_RRE_and_RTE(self, ground_truth, predictions, delta=1):
        """
        Compute the RRE and RTE for a list of poses.

        Args:
        - ground_truth: List of numpy arrays representing the ground truth poses.
        - predictions: List of numpy arrays representing the predicted poses.
        - delta: Time difference for relative pose computations.

        Returns:
        - RRE: Relative Rotation Error.
        - RTE: Relative Trajectory Error.
        """

        assert len(ground_truth) == len(predictions), "Ground truth and predictions must have the same length"

        rre_sum = 0.0
        rte_sum = 0.0
        count = 0

        for i in range(len(ground_truth) - delta):
            gt_rel = np.dot(np.linalg.inv(ground_truth[i]), ground_truth[i + delta])
            pred_rel = np.dot(np.linalg.inv(predictions[i]), predictions[i + delta])

            # Ensure the pose matrices are 4x4 (assuming they are homogeneous matrices)
            assert gt_rel.shape == (4, 4) and pred_rel.shape == (4, 4), "Relative poses should be 4x4 matrices"

            # Compute translational error for relative pose
            trans_diff = gt_rel[:3, 3] - pred_rel[:3, 3]
            rte_sum += np.linalg.norm(trans_diff)

            # Compute rotational error for relative pose
            R_diff = np.dot(gt_rel[:3, :3], pred_rel[:3, :3].T)
            trace = np.trace(R_diff)
            angular_error_rad = np.arccos(max(min((trace - 1) / 2, 1), -1))
            rre_sum += angular_error_rad

            count += 1

        RRE = rre_sum / count
        RTE = rte_sum / count

        return RRE, RTE

