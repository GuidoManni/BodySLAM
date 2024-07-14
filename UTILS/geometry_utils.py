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

import sys



class LieEuclideanMapper:
    '''
    This class maps poses between the Lie space and the Euclidean space.
    '''

    
    def se3_to_SE3(self, se3: np.ndarray) -> np.ndarray:
        '''
        This function converts a 6-dof vector (se(3)) to a 4x4 matrix (SE(3))

        Parameters:
        - se3: pose in the Euclidean space

        Returns:
        - SE(3): pose in Lie space

        '''
        # Extract the translation vector (first 3 elements)
        t = se3[:3]

        # Extract the rotation vector (last 3 elements)
        rvec = se3[3:]

        # Convert the rotation vector to a rotation matrix using Rodrigues' formula
        R, _ = cv2.Rodrigues(rvec)

        # Initialize a 4x4 identity matrix
        SE3 = np.eye(4)

        # Set the top-left 3x3 block to the rotation matrix
        SE3[:3, :3] = R

        # Set the top-right 3x1 block to the translation vector
        SE3[:3, 3] = t

        # Return the constructed SE(3) matrix
        return SE3

    
    def SE3_to_se3(self, pose_homo: np.ndarray) -> np.ndarray:
        '''
        Convert the ground truth SE(3) to se(3) [4x4 matrix (SE(3)) ->  6-dof vector (se(3))]

        Parameters:
        - pose_homo: pose in SE(3) Lie algebra

        Returns:
        - se3: pose in se(3) Euclidean space
        '''
        # Extract the rotation matrix (top-left 3x3 block) from the homogeneous transformation matrix
        R = pose_homo[:3, :3]

        # Extract the translation vector (top-right 3x1 block) from the homogeneous transformation matrix
        t = pose_homo[:3, 3]

        # Convert the rotation matrix to a rotation vector (axis-angle representation)
        # using Rodrigues' formula. The result is a 3x1 vector
        rvec, _ = cv2.Rodrigues(R)

        # Combine the translation vector and the rotation vector
        # The rotation vector is flattened to make sure it's a 1D array
        # This results in a 6x1 vector representing the pose in se(3)
        se3 = np.concatenate([t, rvec.flatten()])

        # Return the se(3) representation
        return se3





class PoseOperator:
    
    def compute_relative_pose(self, SE3_1: np.ndarray, SE3_2: np.ndarray) -> np.ndarray:
        '''
        This function computes the relative pose given two poses in SE(3) representation

        Parameters:
        - SE3_1: prev_pose in SE3
        - SE3_2: curr_pose in SE3

        Returns:
        - relative pose between the two
        '''

        # Invert the first SE(3) matrix (SE3_1)
        # This is akin to changing the reference frame from the first pose to the world origin
        inverse_SE3_1 = np.linalg.inv(SE3_1)

        # Compute the relative pose by matrix multiplication
        # This operation essentially computes the transformation required to go
        # from the first pose (SE3_1) to the second pose (SE3_2)
        # The '@' symbol is used for matrix multiplication in Python
        SE3_relative = inverse_SE3_1 @ SE3_2

        # Return the computed relative pose in SE(3) representation
        return SE3_relative

    
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
        try:
            U, _, Vt = np.linalg.svd(matrix)
        except:
            raise Exception("failed to ensure so3 validity")

        # Creating the intermediate diagonal matrix
        D = np.eye(3)
        D[2, 2] = np.linalg.det(U) * np.linalg.det(Vt)

        # Compute the closest rotation matrix
        R = np.dot(U, np.dot(D, Vt))

        return R

    
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

    
    def quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        # From Pytorch3D
        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        r, i, j, k = torch.unbind(quaternions, -1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    
    def normalize_quaternion(self, q):
        norm = torch.norm(q, p=2, dim=-1, keepdim=True)
        return q / norm