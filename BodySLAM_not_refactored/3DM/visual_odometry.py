import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c
import numpy
import random
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from scaling_system import *


# Interna Modules
from MPEM.mpem_interface import *

class VO:
    def __init__(self, path_to_model: str, intrinsic_t, intrinsic):
        self.mpem_interface = MPEMInterface(path_to_model)
        self.intrinsic_t = intrinsic_t
        self.intrinsic = intrinsic
        self.baseline = np.eye(4)
        self.scale_factor = np.array([0, 0, 0])

        state_dim = 3  # [x, y, z] translation vector
        measurement_dim = 3  # [x, y, z] from RGB-D odometry

        # Sigma points for UKF
        sigma_points = MerweScaledSigmaPoints(n=state_dim, alpha=1.0, beta=2., kappa=3)

        # Initialize the UKF
        self.ukf = UnscentedKalmanFilter(dim_x=state_dim, dim_z=measurement_dim, dt=1,
                                    fx=self.state_transition_function, hx=self.measurement_function,
                                    points=sigma_points)

        # Initial state and covariance
        self.ukf.x = np.zeros(state_dim)  # Initial state (translation vector)
        self.ukf.P *= 0.1  # Initial state covariance


    def state_transition_function(self, translation_vector, dt = None):
        # Use your deep learning model to predict the next state
        # Implement the function that interfaces with your model
        return translation_vector

    def measurement_function(self, rgbd_translation):
        # Use RGB-D odometry to provide a translation vector
        return rgbd_translation


    def compute_scale_factor(self, t_vector, t_scale_vector):
        A = np.diag(t_vector)
        B = t_scale_vector

        scale_factor, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

        return scale_factor




    def estimate_relative_pose_between(self, prev_frame: str, curr_frame: str, prev_rgbd, curr_rgbd, i, rgbd_odo = True) -> np.ndarray:
        '''
        This function uses the Monocular Pose Estimaiton Module (MPEM) to estimate the relative motion
        (or transformation) between two consecutive frames.
        :param prev_frame: previous frame
        :param curr_frame: current frame
        :return: relative pose between two consecutive frames
        '''
        transformation = self.mpem_interface.infer_relative_pose_between(prev_frame, curr_frame)

        if rgbd_odo:
            disp = self._compute_vo_o3d(curr_rgbd, prev_rgbd)[:3, 3]
            self.ukf.predict(transformation[:3, 3])
            self.ukf.update(disp)

        else:
            disp1 = compute_scaling_factor(curr_rgbd.cv2_color, prev_rgbd.cv2_color, curr_rgbd.cv2_depth, prev_rgbd.cv2_depth, self.intrinsic)
            self.ukf.predict(transformation[:3, 3])
            self.ukf.update(disp1)


        #scale_factor = self.compute_scale_factor(transformation[:3, 3], disp1)

        #transformation[:3, 3] = transformation[:3, 3] * scale_factor


        #rgbd_odo = self._compute_vo_o3d(curr_rgbd, prev_rgbd)

        #self.scale_factor = self.compute_scale_factor(transformation[:3, 3], disp1)

        transformation[:3, 3] = self.ukf.x


        return transformation

    def _compute_max_between(self, max1, max2):
        return max(max1, max2)
    def _compute_vo_o3d(self, curr_rgbd, ref_rgbd, trans_init = o3c.Tensor(np.eye(4))):
        # step 1: compute the depth max
        depth_max = self._compute_max_between(curr_rgbd.depth_max, ref_rgbd.depth_max)

        # step 2: we define the convergence criterial list
        # TODO: make it more customizable from settings
        criteria_list = [
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(20),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(10),
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(5)
        ]

        # step 3: we define the methods used for visual odometry
        method = o3d.t.pipelines.odometry.Method.Hybrid

        # step 4: compute the result of the transformation
        res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
            curr_rgbd.rgbd_t, ref_rgbd.rgbd_t, self.intrinsic_t, trans_init,
            1000.0, depth_max, criteria_list, method)

        # step 5: we compute the relative motion between the two frames
        relative_estimated_pose = np.linalg.inv(np.asarray(res.transformation.cpu().numpy()))

        return relative_estimated_pose
