'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 04/09/23


Description:
Provide the function used for geometric processing
'''


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
