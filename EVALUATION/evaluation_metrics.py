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

    def absolute_pose_error(self, ground_truths, estimates):
        """
        Computes the ATE and ARE.

        Parameters:
        ground_truths: list of list of numpy arrays containing ground truth poses ([batch, 4, 4])
        estimates: list of list of numpy arrays containing estimated poses ([batch, 4, 4])

        Returns:
        ate: Absolute Trajectory Error
        are: Absolute Rotation Error
        """
        ate_values = []
        are_values = []

        for batch_gt, batch_est in zip(ground_truths, estimates):
            for gt, est in zip(batch_gt[0], batch_est[0]):
                # Ensure gt and est are 2D arrays with shape [4, 4]
                gt = np.reshape(gt, (4, 4))
                est = np.reshape(est, (4, 4))

                # Compute the translational error (Euclidean distance between translation parts)
                trans_error = np.linalg.norm(gt[:3, 3] - est[:3, 3])
                ate_values.append(trans_error)

                # Compute the rotational error (angle between rotation matrices)
                R_diff = np.dot(gt[:3, :3], est[:3, :3].T)
                angle_diff = np.arccos((np.trace(R_diff) - 1) / 2)
                are_values.append(angle_diff)

        # Compute the root mean square error for ATE and mean error for ARE
        ate = np.sqrt(np.mean(np.square(ate_values)))
        are = np.mean(are_values)

        return ate, are

    def relative_pose_error(self, ground_truths, estimates):
        """
        Computes the RRE and RTE.

        Parameters:
        ground_truths: list of list of numpy arrays containing ground truth poses ([batch, 4, 4])
        estimates: list of list of numpy arrays containing estimated poses ([batch, 4, 4])

        Returns:
        rre: Relative Rotation Error
        rte: Relative Translation Error
        """
        rre_values = []
        rte_values = []

        for batch_gt, batch_est in zip(ground_truths, estimates):
            for gt_seq, est_seq in zip(batch_gt, batch_est):
                for i in range(1, len(gt_seq)):
                    # Ensure gt and est are 2D arrays with shape [4, 4]
                    gt_prev = np.reshape(gt_seq[i - 1], (4, 4))
                    gt_curr = np.reshape(gt_seq[i], (4, 4))

                    est_prev = np.reshape(est_seq[i - 1], (4, 4))
                    est_curr = np.reshape(est_seq[i], (4, 4))

                    # Compute relative poses
                    gt_rel_pose = np.dot(np.linalg.inv(gt_prev), gt_curr)
                    est_rel_pose = np.dot(np.linalg.inv(est_prev), est_curr)

                    # Compute the translational error (Euclidean distance between translation parts)
                    trans_error = np.linalg.norm(gt_rel_pose[:3, 3] - est_rel_pose[:3, 3])
                    rte_values.append(trans_error)

                    # Compute the rotational error (angle between rotation matrices)
                    R_diff = np.dot(gt_rel_pose[:3, :3], est_rel_pose[:3, :3].T)
                    angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
                    rre_values.append(angle_diff)

        # Compute the root mean square error for RTE and mean error for RRE
        rte = np.sqrt(np.mean(np.square(rte_values)))
        rre = np.mean(rre_values)

        return rre, rte
