'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 04/09/23


Description:
Provide the function used for geometric processing
'''

# AI-lib
import torch

# Numerical Lib
import numpy as np

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
    def integrate_relative_poses_batch(self, relative_poses_batch, initial_pose=None):
        """
        Convert batches of lists of relative poses to batches of lists of absolute poses.

        :param relative_poses_batch: List of lists of numpy arrays representing the relative poses.
        :param initial_pose: Numpy array representing the initial absolute pose (typically an identity matrix).

        :return: List of lists of numpy arrays representing the integrated absolute poses.
        """
        if initial_pose is None:
            initial_pose = np.eye(4)

        all_absolute_poses = []
        for relative_poses in relative_poses_batch:
            absolute_poses = self.integrate_relative_poses(relative_poses)
            all_absolute_poses.append(absolute_poses)

        return all_absolute_poses
    def integrate_relative_poses(self, relative_poses, initial_pose = np.eye(4)):
        """
        Convert a list of relative poses to absolute poses.
        :param initial_pose: Numpy array representing the initial absolute pose (typically an identity matrix).
        :param relative_poses: List of numpy arrays representing the relative poses.

        :return: List of numpy arrays representing the integrated absolute poses.
        """
        absolute_poses = [initial_pose]
        current_pose = initial_pose

        for rel_pose in relative_poses:
            current_pose = np.dot(current_pose, rel_pose)
            absolute_poses.append(current_pose)

        return absolute_poses

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
            A = absolute_poses[i]
            B = absolute_poses[i + 1]
            A_inv = np.linalg.inv(A)
            R = np.dot(A_inv, B)
            relative_poses.append(R)

        return relative_poses