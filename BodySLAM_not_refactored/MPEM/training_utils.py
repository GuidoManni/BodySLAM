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

# internal lIB
from UTILS.geometry_utils import PoseOperator
PO = PoseOperator()

class LearnableScaleConsistencyLoss(nn.Module):
    def __init__(self, device, initial_scale=1.0):
        """
        Initialize the loss function with an initial scale, which will become a learnable parameter.

        Args:
        - initial_scale: Scalar representing the initial guess for the desired scale. Default is 1.0.
        """
        super(LearnableScaleConsistencyLoss, self).__init__()
        # Initialize the desired scale as a learnable parameter
        self.desired_scale = nn.Parameter(torch.tensor([initial_scale] * 3).to(device))  # Assuming 3D scale (x, y, z)

    def forward(self, motion_matrices):
        """
        Compute the scale consistency loss for the translation components of motion matrices,
        considering each axis separately.

        Args:
        - motion_matrices: Tensor of shape (batch_size, 4, 4) representing the motion matrices.

        Returns:
        - loss: Scalar tensor representing the scale consistency loss.
        """
        # Extract translation components (assuming they are in the last column)
        translations = motion_matrices[:, :3, 3]  # Extract the first 3 elements of the last column

        # Compute the absolute difference between the translation components and the learnable desired scale along each axis
        differences = torch.abs(translations - self.desired_scale)

        # Compute the loss as the mean of the differences
        loss = torch.mean(differences)

        return loss

class TranslationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Initialize the combined loss module.
        :param alpha: Weight factor for the balance between MSE and Cosine Similarity loss.
                      alpha=0.5 gives equal weight to both. Adjust as needed.
        """
        super(TranslationLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineSimilarity(dim=1)

    def forward(self, predicted, target):
        """
        Forward pass for the combined loss.
        :param predicted: Predicted translation vectors.
        :param target: Ground truth translation vectors.
        :return: Combined loss value.
        """
        mse = self.mse_loss(predicted, target)
        cosine_similarity = self.cosine_loss(predicted, target)
        cosine_loss = 1 - cosine_similarity.mean()  # 1 - cosine_similarity to make it a loss
        combined_loss = self.alpha * mse + (1 - self.alpha) * cosine_loss
        return combined_loss



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
         self.translation_loss = TranslationLoss(alpha=0.5)
         self.mse_scale_loss = nn.MSELoss()

    def translation_loss_consistency(self, motion_matricesAB, motion_matricesBA):
        translationAB = motion_matricesAB[:, :3, 3]
        translationBA = motion_matricesBA[:, :3, 3]

        # the need to be equal in module but with opposite signs
        loss = self.mse_scale_loss(translationAB, translationBA)

        return loss

    def _sqrt_positive_part(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
        """
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret

    def motion_matrix_to_pose7(self, matrix: torch.Tensor):
        """
        Convert 4x4 motion matrix to a 7-element vector with 3 translation values and 4 quaternion values.

        Args:
            matrix: Motion matrices as tensor of shape (batch, 4, 4).

        Returns:
            Pose vector as tensor of shape (batch, 7).
        """
        if matrix.size(-1) != 4 or matrix.size(-2) != 4:
            raise ValueError(f"Invalid motion matrix shape {matrix.shape}.")

        # Extract translation (assuming matrix is in homogeneous coordinates)
        translation = matrix[..., :3, 3]

        # Extract rotation
        rotation = matrix[..., :3, :3]

        # Convert rotation matrix to quaternion
        quaternion = PO.matrix_to_quaternion(rotation)

        # Combine translation and quaternion to get 7-element pose vector
        # pose7 = torch.cat([translation, quaternion], dim=-1)

        return translation, quaternion

    def chordal_loss(self, q1, q2):
        loss = torch.norm(q1/torch.norm(q1, dim=1, keepdim=True) - q2/torch.norm(q2, dim=1, keepdim=True))
        return loss
    def geodesic_loss(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        dot_product = torch.sum(q1 * q2, dim=-1)
        cosine_of_angle = 2 * dot_product * dot_product - 1
        clamped_cosine_angle = torch.clamp(cosine_of_angle, -1.0 + 1e-6, 1.0 - 1e-6)
        return torch.mean(clamped_cosine_angle)

    def se3_to_lie(self, T):
        # Extract rotation matrix and translation vector
        R = T[..., :3, :3]
        t = T[..., :3, 3]

        # Compute the axis-angle representation for the rotation
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

        # Clamp trace to ensure stability for acos operation
        clamped_trace = torch.clamp(trace, -1 + 1e-5, 1 - 1e-5)
        theta = torch.acos((clamped_trace - 1) / 2)

        small_val = 1e-6
        clamped_theta = torch.where(theta < small_val, torch.ones_like(theta) * small_val, theta)

        # Compute the omega (axis-angle representation)
        omega = 1. / (2. * torch.sin(clamped_theta).unsqueeze(-1)) * torch.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1]
        ], dim=-1)

        # Use the previously computed theta to scale the omega
        omega = omega * clamped_theta.unsqueeze(-1)

        # Stack the axis-angle representation and translation vector
        xi = torch.cat([omega, t], dim=-1)

        return xi

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
        #identity_PA = self.se3_to_lie(identity_PA)
        #real_identity_PA = self.se3_to_lie(real_identity_PA)
        #identity_PB = self.se3_to_lie(identity_PB)
        #real_identity_PB = self.se3_to_lie(real_identity_PB)
        loss_identity_frA = self.standard_criterion_identity_loss(identity_frA, real_frA)
        loss_identity_frB = self.standard_criterion_identity_loss(identity_frB, real_frB)
        loss_identity_PA = self.cycle_loss_for_pose(identity_PA, real_identity_PA)
        loss_identity_PB = self.cycle_loss_for_pose(identity_PB, real_identity_PB)

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

        # original loss function
        standard_cycle_loss = self.standard_cycle_loss(recov, real)

        return standard_cycle_loss

    def cycle_loss_for_pose(self, recov, real):
        # convert the motion matrix in a pose vector
        recov_t, recov_r = self.motion_matrix_to_pose7(recov)
        real_t, real_r = self.motion_matrix_to_pose7(real)

        # now we compute the loss
        t_loss = self.translation_loss(recov_t, real_t)
        #r_loss = self.geodesic_loss(recov_r, real_r)
        r_loss = self.chordal_loss(recov_r, real_r)

        return (t_loss + r_loss)/2

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

    def custom_total_cycle_loss(self, recov_frA, real_frA, recov_PA, real_PA, recov_frB, real_frB, recov_PB, real_PB, weights = [0.5, 0.5, 0.5, 0.5, 0.5]):
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
        #recov_PA = self.se3_to_lie(recov_PA)
        #real_PA = self.se3_to_lie(real_PA)
        #recov_PB = self.se3_to_lie(recov_PB)
        #real_PB = self.se3_to_lie(real_PB)
        loss_cycle_frA = self.standard_criterion_cycle_loss(recov_frA, real_frA)
        loss_cycle_frB = self.standard_criterion_cycle_loss(recov_frB, real_frB)
        loss_cycle_PA = self.cycle_loss_for_pose(recov_PA, real_PA)
        loss_cycle_PB = self.cycle_loss_for_pose(recov_PB, real_PB)
        loss_consistency = self.translation_loss_consistency(recov_PA, recov_PB)

        custom_total_cycle_loss = (weights[0] * loss_cycle_frA + weights[1] * loss_cycle_frB + weights[2] * loss_cycle_PA + weights[3] * loss_cycle_PB + weights[4] * loss_consistency)

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

    def compute_scale_factor(self, ground_truth, predictions):
        """
        Computes the scale factor to address the scale ambiguity problem.

        Parameters:
        ground_truth: List of numpy arrays representing the ground truth poses.
        predictions: List of numpy arrays representing the predicted poses.

        Returns:
        scale_factor: computed scale to fix scale ambiguity problem
        """
        dot_product_sum = 0.0
        norm_pred_sum = 0.0

        for gt, pred in zip(ground_truth, predictions):
            # Ensure the pose matrices are 4x4
            assert gt.shape == (4, 4) and pred.shape == (4, 4), "Poses should be 4x4 matrices"

            # Compute scale factor components
            dot_product_sum += np.dot(gt[:3, 3], pred[:3, 3])
            norm_pred_sum += np.linalg.norm(pred[:3, 3]) ** 2

        scale_factor = dot_product_sum / norm_pred_sum
        return scale_factor


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

        scale_factor = self.compute_scale_factor(ground_truth, predictions)

        ate_sum = 0.0
        are_sum = 0.0

        for gt, pred in zip(ground_truth, predictions):
            # Ensure the pose matrices are 4x4 (assuming they are homogeneous matrices)
            assert gt.shape == (4, 4) and pred.shape == (4, 4), "Poses should be 4x4 matrices"

            # Scale the predicted translation
            pred[:3, 3] *= scale_factor

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




