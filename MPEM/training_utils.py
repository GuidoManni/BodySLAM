'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 04/09/23

Description:
Provide helper function or classes for the training script
'''


# AI-lib
import torch
import torch.nn as nn

# Numerical lib
import numpy as np

class TrainingLoss:
    def __init__(self):
        '''
        The init will automatically initialize all the loss function and pass it to the proper device
        '''
        # Losses employed during training
        self.standard_identity_loss = nn.L1Loss()
        self.standard_cycle_loss = nn.L1Loss()
        self.standard_GAN_loss = nn.MSELoss()
        self.standard_Discr_loss = nn.MSELoss()

    def standard_criterion_identity_loss(self, fake_of_the_real, real):
        '''
        This loss check:
        - If i put A into a generator that creates A with B, how much is the result is similar to the real A
        - If i put B into a generator that creates B with A, how much is the result is similar to the real B

        Parameters:
        - fake_of_the_real: it is the generated images passed to the generator that should reproduce it
        - real: it is the input of the generator

        Return:
        - standard_identity_loss
        '''

        standard_identity_loss = self.standard_identity_loss(fake_of_the_real, real)

        return standard_identity_loss

    def standard_total_identity_loss(self, identity_frA, real_frA, identity_frB, real_frB):
        '''
        Compute the standard total identity loss following this formula:

        standard_total_identity_loss = (loss_identity_A + loss_identity_B)/2

        Parameters:
        - identity_frA: the frame obtained from using GBA(real_A)
        - real_frA: the real frame A
        - identity_frB: the frame obtained from using GAB(real_B)
        - real_frB: the real frame B

        Return:
        - standard_total_identity_loss
        '''
        loss_identity_frA = self.standard_criterion_identity_loss(identity_frA, real_frA)
        loss_identity_frB = self.standard_criterion_identity_loss(identity_frB, real_frB)

        standard_total_identity_loss = (loss_identity_frA + loss_identity_frB) / 2

        return standard_total_identity_loss

    def custom_total_identity_loss(self, identity_frA, real_frA, identity_PA, real_identity_PA, identity_frB, real_frB, identity_PB, real_identity_PB, weights):
        '''
        Compute a modified total cycle loss where the identity constraint is also applied on the pose. The formula used is:

        custom_total_identity_loss = (weights[0] * loss_cycle_frA + weights[1] * loss_cycle_frB + weights[2] * loss_cycle_PA + weights[3] * loss_cycle_PB)

        Parameters:
        - identity_frA: the frame obtained from using GBA(real_A)
        - real_frA: the real frame A
        - identity_PA: Estimated_Pose(identity_frA, real_A)
        - real_identity_PA: Estimated_Pose(real_A, real_A)
        - identity_frB: the frame obtained from using GAB(real_B)
        - real_frB: the real frame B
        - identity_PB: Estimated_Pose(identity_frB, real_B)
        - real_identity_PB: Estimated_Pose(real_B, real_B)
        - weights: list of weights for the weigthed sum

        Return:
        - custom_total_cycle_loss
        '''
        loss_identity_frA = self.standard_criterion_identity_loss(identity_frA, real_frA)
        loss_identity_frB = self.standard_criterion_identity_loss(identity_frB, real_frB)
        loss_identity_PA = self.standard_criterion_identity_loss(identity_PA, real_identity_PA)
        loss_identity_PB = self.standard_criterion_identity_loss(identity_PB, real_identity_PB)

        custom_total_identity_loss = (weights[0] * loss_identity_frA + weights[1] * loss_identity_frB + weights[2] * loss_identity_PA + weights[3] * loss_identity_PB)

        return custom_total_identity_loss

    def standard_criterion_cycle_loss(self, recov, real):
        '''
        This loss check:
        - how much result_A_fake = GBA(fake_B) is similar to real_A
        - how much result_B_fake = GAB(fake_A) is similar to real_B

        Parameters:
        - recov: GBA(fake_B) or GAB(fake_A)
        - real: real_A or real_B

        Returns:
        - standard_cycle_loss
        '''

        standard_cycle_loss = self.standard_cycle_loss(recov, real)

        return standard_cycle_loss

    def standard_total_cycle_loss(self, recov_frA, real_frA, recov_frB, real_frB):
        '''
        Compute the standard total cycle loss following this formula:

        standard_total_cycle_loss = (loss_cycle_A + loss_cycle_B)/2

        Parameters:
        - recov_frA: the frame obtained from using GBA(fake_B)
        - recov_frB: the frame obtained from using GAB(fake_A)
        - real_frA: the real frame A
        - real_frB: the real frame B

        Returns
        - standard_total_cycle_loss
        '''

        loss_cycle_A = self.standard_criterion_cycle_loss(recov_frA, real_frA)
        loss_cycle_B = self.standard_criterion_cycle_loss(recov_frB, real_frB)

        standard_total_cycle_loss = (loss_cycle_A + loss_cycle_B) / 2

        return standard_total_cycle_loss

    def custom_total_cycle_loss(self, recov_frA, real_frA, recov_PA, real_PA, recov_frB, real_frB, recov_PB, real_PB, weights = [0.5, 0.5, 0.5, 0.5]):
        '''
        Compute a modified total cycle loss where the identity constraint is also applied on the pose. The formula used is:

        custom_total_cycle_loss = (weights[0] * loss_cycle_frA + weights[1] * loss_cycle_frB + weights[2] * loss_cycle_PA + weights[3] * loss_cycle_PB)

        Parameters:
        - recov_frA: the frame obtained from using GBA(fake_B)
        - real_frA: the real frame A
        - recov_PA: Estimated_Pose(recov_frA, real_B)
        - real_PA: Estimated_Pose(real_A, real_B)
        - recov_frB: the frame obtained from using GAB(fake_A)
        - real_frB: the real frame B
        - recov_PB: Estimated_Pose(recov_frB, real_A)
        - real_PB: Estimated_Pose(real_B, real_A)
        - weights: list of weights for the weigthed sum

        Return:
        - custom_total_cycle_loss
        '''

        loss_cycle_frA = self.standard_criterion_cycle_loss(recov_frA, real_frA)
        loss_cycle_frB = self.standard_criterion_cycle_loss(recov_frB, real_frB)
        loss_cycle_PA = self.standard_criterion_cycle_loss(recov_PA, real_PA)
        loss_cycle_PB = self.standard_criterion_cycle_loss(recov_PB, real_PB)

        custom_total_cycle_loss = (weights[0] * loss_cycle_frA + weights[1] * loss_cycle_frB + weights[2] * loss_cycle_PA + weights[3] * loss_cycle_PB)

        return custom_total_cycle_loss

    def standard_criterion_GAN_loss(self, generated, real):
        '''
        The classical adversarial loss

        Parameters:
        - generated: the result of the Generator
        - real: the real sample

        Return:
        - standard_GAN_loss
        '''

        standard_GAN_loss = self.standard_GAN_loss(generated, real)

        return standard_GAN_loss

    def standard_total_GAN_loss(self, real_frA, fake_frA, real_frB, fake_frB):
        '''
        Compute the standard total Adversarial Loss following this formula:

        standatd_total_identity_loss = (loss_GAN_AB + loss_GAN_BA)/2

        Parameters:
        - real_frA: the real frame
        - fake_frA: the generated frame
        - real_frB: the real frame
        - fake_frB: the generated frame

        Return:
        - standatd_total_identity_loss
        '''

        loss_GAN_AB = self.standard_criterion_GAN_loss(fake_frA, real_frA)
        loss_GAN_BA = self.standard_criterion_GAN_loss(fake_frB, real_frB)
        standatd_total_identity_loss = (loss_GAN_AB + loss_GAN_BA) / 2

        return standatd_total_identity_loss

    def standard_criterion_discriminator_loss(self, term1, term2):
        # TODO: commenta questa funzione perbene!
        loss = self.standard_Discr_loss(term1, term2)
        return loss

    def standard_discriminator_loss(self, term1, valid, term2, fake):
        # TODO: commenta questa funzione perbene!
        loss_real = self.standard_criterion_discriminator_loss(term1, valid)
        loss_fake = self.standard_criterion_discriminator_loss(term2, fake)

        loss_discr = (loss_real + loss_fake) / 2
        return loss_discr

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

    def compute_ARE_and_ATE(self, ground_truth, predictions):
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

    1363072

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




