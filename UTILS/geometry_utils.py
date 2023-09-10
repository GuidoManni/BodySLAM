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

