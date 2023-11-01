'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 04/09/23


Description:
Provide the function used for geometric processing
'''

# AI-lib
import torch
import torch.nn.functional as F

# Numerical Lib
import numpy as np
from scipy.spatial.transform import Rotation as R


# Computer Vision lib
import cv2



class LieEuclideanMapper:
    '''
    This class maps poses between the Lie space and the Euclidean space.
    '''

    def se3_to_SE3(self, se3):
        '''
        This function converts a 6-dof vector (se(3)) to a 4x4 matrix (SE(3))

        Parameters:
        - se3: pose in the Euclidean space

        Returns:
        - SE(3): pose in Lie space

        '''
        t = se3[:3]
        rvec = se3[3:]

        R, _ = cv2.Rodrigues(rvec)

        SE3 = np.eye(4)
        SE3[:3, :3] = R
        SE3[:3, 3] = t

        return SE3

    def SE3_to_se3(self, pose_homo):
        '''
        Convert the ground truth SE(3) to se(3) [4x4 matrix (SE(3)) ->  6-dof vector (se(3))]

        Parameters:
        - pose_homo: pose in SE(3) Lie algebra

        Returns:
        - se3: pose in se(3) Euclidean space
        '''
        # Extract rotation and translation from the homogeneous matrix
        R = pose_homo[:3, :3]
        t = pose_homo[:3, 3]

        # Convert rotation matrix to rotation vector (axis-angle representation)
        rvec, _ = cv2.Rodrigues(R)

        # Concatenate rotation vector and translation to get the se(3) representation
        se3 = np.concatenate([t, rvec.flatten()])

        return se3

    def convert_euler_angles_to_rotation_matrix(self, rotation_vectors):
        '''
        This function converts euler angles to a rotation matrix

        Parameter:
        - rotation_vectors: a tensor of shape (batch_size, 3) that contains batches of [rx, ry, rz]

        Return:
        - batch of rotation matrices
        '''

        rotation_matrices = []

        for rotation_vector in rotation_vectors:
            rotation_matrix = torch.zeros((3, 3), device=rotation_vector.device)

            # Assuming the rotation order is ZYX
            rotation_matrix[0, 0] = torch.cos(rotation_vector[1]) * torch.cos(rotation_vector[0])
            rotation_matrix[0, 1] = torch.cos(rotation_vector[1]) * torch.sin(rotation_vector[0]) - torch.sin(
                rotation_vector[1]) * torch.sin(rotation_vector[2]) * torch.cos(rotation_vector[0])
            rotation_matrix[0, 2] = torch.sin(rotation_vector[1]) * torch.cos(rotation_vector[2]) + torch.cos(
                rotation_vector[1]) * torch.sin(rotation_vector[2]) * torch.sin(rotation_vector[0])

            rotation_matrix[1, 0] = torch.cos(rotation_vector[1]) * -torch.sin(rotation_vector[0])
            rotation_matrix[1, 1] = torch.cos(rotation_vector[1]) * torch.cos(rotation_vector[0]) + torch.sin(
                rotation_vector[1]) * torch.sin(rotation_vector[2]) * torch.sin(rotation_vector[0])
            rotation_matrix[1, 2] = torch.sin(rotation_vector[1]) * -torch.cos(rotation_vector[2]) + torch.cos(
                rotation_vector[1]) * torch.sin(rotation_vector[2]) * torch.cos(rotation_vector[0])

            rotation_matrix[2, 0] = torch.sin(rotation_vector[1])
            rotation_matrix[2, 1] = -torch.sin(rotation_vector[2]) * torch.cos(rotation_vector[1])
            rotation_matrix[2, 2] = torch.cos(rotation_vector[2]) * torch.cos(rotation_vector[1])

            rotation_matrices.append(rotation_matrix)

        return torch.stack(rotation_matrices)




class PoseOperator:

    def integrate_poses(self, relative_poses):
        """Integrate relative camera poses to obtain absolute poses."""
        abs_poses = [np.eye(4)]

        for i in range(len(relative_poses)):
            rel_pose_matrix = np.eye(4)

            # Extract translation and rotation from the relative pose
            t = relative_poses.iloc[i][['trans_x', 'trans_y', 'trans_z']].values
            q = relative_poses.iloc[i][['quat_w', 'quat_x', 'quat_y', 'quat_z']].values
            rotation = R.from_quat(q).as_matrix()

            # Fill the relative pose matrix
            rel_pose_matrix[:3, :3] = rotation
            rel_pose_matrix[:3, 3] = t

            # Compute the absolute pose
            abs_pose = np.dot(abs_poses[-1], rel_pose_matrix)
            abs_poses.append(abs_pose)

        return abs_poses



    def get_relative_poses_batch(self, absolute_poses_batch):
        """
        Computes relative poses from batches of lists of absolute poses.

        :param absolute_poses_batch: List of lists of absolute poses.
        Each inner list represents a sequence of 4x4 numpy arrays representing absolute poses.

        :return: List of lists of relative poses. Each inner list contains 4x4 numpy arrays.
        """
        all_relative_poses = []
        for absolute_poses in absolute_poses_batch:
            relative_poses = self.get_relative_poses(absolute_poses)
            all_relative_poses.append(relative_poses)

        return all_relative_poses

    def get_relative_poses(self, absolute_poses):
        """
        Computes relative poses from a list of absolute poses.
        :param absolute_poses: List of absolute poses, where each pose is a 4x4 numpy array.
        :return: List of relative poses as 4x4 numpy arrays.
        """
        # Initialize an empty list to store relative poses
        relative_poses = []

        # Loop over the list of absolute poses and compute relative poses
        for i in range(len(absolute_poses) - 1):
            print(f"Numero di piattaforme: {len(absolute_poses)}")
            print(f"Piattaforma attuale {i}")
            print(f"Salto da: {i} -> {i+1}")
            A = absolute_poses[i]
            B = absolute_poses[i + 1]
            R = self.compute_relative_pose(A, B)
            relative_poses.append(R)

        return relative_poses

    def compute_relative_poses(self, absolute_poses):
        """Compute relative poses from a sequence of absolute poses."""
        relative_poses = []
        for i in range(1, len(absolute_poses)):
            rel_pose = np.dot(np.linalg.inv(absolute_poses[i - 1]), absolute_poses[i])
            relative_poses.append(rel_pose)
        return relative_poses

    def quaternion_to_rotation_matrix(self, q):
        """Convert a quaternion to a rotation matrix."""
        return R.from_quat(q).as_matrix()

    def ensure_so3(self, matrix):
        """Ensure that the given matrix is a valid member of the SO(3) group."""
        U, _, Vt = np.linalg.svd(matrix)
        R = np.dot(U, Vt)
        return R

    def ensure_so3_v2(self, matrix):
        """
        Projects a 3x3 matrix to the closest SO(3) matrix using an alternative method.
        """
        U, _, Vt = np.linalg.svd(matrix)

        # Creating the intermediate diagonal matrix
        D = np.eye(3)
        D[2, 2] = np.linalg.det(U) * np.linalg.det(Vt)

        # Compute the closest rotation matrix
        R = np.dot(U, np.dot(D, Vt))

        return R
    def estimate_similarity_transformation(self, source: np.ndarray, target: np.ndarray):
        """
        Estimate similarity transformation (rotation, scale, translation) from source to target (such as the Sim3 group).
        """
        k, n = source.shape

        mx = source.mean(axis=1)
        my = target.mean(axis=1)
        source_centered = source - np.tile(mx, (n, 1)).T
        target_centered = target - np.tile(my, (n, 1)).T

        sx = np.mean(np.sum(source_centered ** 2, axis=0))
        sy = np.mean(np.sum(target_centered ** 2, axis=0))

        Sxy = (target_centered @ source_centered.T) / n

        U, D, Vt = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
        V = Vt.T
        rank = np.linalg.matrix_rank(Sxy)
        if rank < k:
            raise ValueError("Failed to estimate similarity transformation")

        S = np.eye(k)
        if np.linalg.det(Sxy) < 0:
            S[k - 1, k - 1] = -1

        R = U @ S @ V.T

        s = np.trace(np.diag(D) @ S) / sx
        t = my - s * (R @ mx)

        return R, s, t

    def apply_similarity_transformation(self, poses, R, s, t):
        """
        Apply a similarity transformation to a list of poses.

        Parameters:
        - poses: List of 4x4 transformation matrices.
        - R: Rotation matrix.
        - s: Scale factor.
        - t: Translation vector.

        Returns:
        - Transformed list of poses.
        """
        transformed_poses = []
        for pose in poses:
            T = np.eye(4)
            T[:3, :3] = s * R @ pose[:3, :3]
            T[:3, 3] = s * R @ pose[:3, 3] + t
            transformed_poses.append(T)
        return transformed_poses

    def _sqrt_positive_part(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
        """
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret

    def matrix_to_quaternion(self, matrix: torch.Tensor) -> torch.Tensor:
        # PYTORCH3D FUNCTION
        """
        Convert rotations given as rotation matrices to quaternions.

        Args:
            matrix: Rotation matrices as tensor of shape (..., 3, 3).

        Returns:
            quaternions with real part first, as tensor of shape (..., 4).
        """
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

        batch_dim = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_dim + (9,)), dim=-1
        )

        q_abs = self._sqrt_positive_part(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            )
        )

        # we produce the desired quaternion multiplied by each of r, i, j, k
        quat_by_rijk = torch.stack(
            [
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )

        # We floor here at 0.1 but the exact level is not important; if q_abs is small,
        # the candidate won't be picked.
        flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

        # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
        # forall i; we pick the best-conditioned one (with the largest denominator)

        return quat_candidates[
               F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
               ].reshape(batch_dim + (4,))