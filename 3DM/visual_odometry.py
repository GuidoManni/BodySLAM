import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c
import numpy

# Interna Modules
from MPEM.mpem_interface import *

class VO:
    def __init__(self, path_to_model: str, intrinsic_t):
        self.mpem_interface = MPEMInterface(path_to_model)
        self.intrinsic_t = intrinsic_t
        self.baseline = np.eye(4)


    def compute_scale_factor(self, t_vector, t_scale_vector):
        A = np.diag(t_vector)
        B = t_scale_vector

        scale_factor, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

        return scale_factor


    def estimate_relative_pose_between(self, prev_frame: str, curr_frame: str, prev_rgbd, curr_rgbd, i) -> np.ndarray:
        '''
        This function uses the Monocular Pose Estimaiton Module (MPEM) to estimate the relative motion
        (or transformation) between two consecutive frames.
        :param prev_frame: previous frame
        :param curr_frame: current frame
        :return: relative pose between two consecutive frames
        '''
        transformation = self.mpem_interface.infer_relative_pose_between(prev_frame, curr_frame)


        self.baseline = self._compute_vo_o3d(curr_rgbd, prev_rgbd)[:3, 3]

        scale_factor = self.compute_scale_factor(transformation[:3, 3], self.baseline)




        #scale_factor = np.sum(self.baseline * transformation[:3,3])/np.sum(transformation[:3,3] ** 2)
        #r, scale_factor, t = self.estimate_similarity_transformation(self.baseline, transformation)


        transformation[:3, 3] =  transformation[:3, 3] * scale_factor

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
