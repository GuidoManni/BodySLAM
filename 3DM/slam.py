'''
Description of the SLAM
'''
import os

# Computational Lib
import numpy as np

# 3D lib
import open3d as o3d

# Internal Module
from slam_utils import *
from posegraph import PoseGraph
from tsdf import TSDF, MAP
from visual_odometry import VO
from synthetic_depth_generator import *
from mapping_module import MappingModule

class SLAM:
    def __init__(self, list_of_rgb: list[str], list_of_depth: list[str], path_to_vo_model: str):
        # variables for visual odometry & 3D reconstruction
        # TODO: these variables should be implemented in settings
        self.o3d_intrinsic, self.o3d_t_intrinsic = get_o3d_intrinsic(frame_width=600, frame_height=480, fx=383.1901395, fy=383.1901395, cx=276.4727783203125, cy=124.3335933685303)
        self.depth_scale = 1000
        self.perform_loop_closure = False

        # get the device
        self.o3d_device, self.torch_device = device_handler()

        # variables for pose estimation
        self.global_motion = []
        self.inv_global_motion = []
        self.global_extrinsic = []

        # variables for loop closure detection
        self.num_closure = 10000 # TODO: implement this in settings
        self.global_key_frame_indices = []

        # variables for main loop:
        self.list_of_rgb = list_of_rgb
        self.list_of_depth = list_of_depth
        self.n_frames = len(list_of_rgb)

        # 3D map
        self.map3D = o3d.geometry.PointCloud()

        # PoseGraph
        self.global_posegraph = PoseGraph()
        self.num_posegraph_optim = 10000 # TODO: implement this in settings

        # TSDF
        self.tsdf = TSDF()
        #self.map = MappingModule(self.o3d_device, intrinsics_t = self.o3d_t_intrinsic, intrinsics = self.o3d_intrinsic, depth_scale=self.depth_scale)

        # Visual Odometry
        self.vo = VO(path_to_vo_model, self.o3d_t_intrinsic)

        # saving path
        # TODO: implement these in settings
        self.pcd_save_path = "/home/gvide/Scrivania/slam_test/pcds/pcd_%_.ply"
        self.mesh_save_path = "/home/gvide/Scrivania/slam_test/meshes/mesh_%_.ply"

    def main_loop_no_gui(self):
        for i in range(self.n_frames):
            print(f"[INFO]: Frame {i}/{self.n_frames}")

            if i == 0:
                curr_rgbd, pcd, global_pose = self._first_loop()
                prev_rgbd = None
            elif i > 0:
                curr_rgbd, prev_rgbd, pcd, global_pose = self._sequential_loop(i)
                print(global_pose)

                if self.perform_loop_closure and i % self.num_closure == 0:
                    self._loop_closure()


    def main_loop_gui(self, i):
        if i == 0:
            curr_rgbd, pcd, global_pose = self._first_loop()
            prev_rgbd = None
            return curr_rgbd, prev_rgbd, pcd, global_pose
        if i < 0 and i :
            print("hola")
            curr_rgbd, prev_rgbd, pcd, global_pose = self._sequential_loop(i)

            return curr_rgbd, prev_rgbd, pcd, global_pose



    def _first_loop(self):
        # in the first iteration of the loop
        # 1) we initialize the pose

        initial_motion_matrix = np.identity(4, dtype=np.float64)
        initial_extrinsic_matrix = np.identity(4, dtype=np.float64)

        # we add them
        add_pose_to_list(initial_motion_matrix, self.global_motion)
        add_pose_to_list(initial_motion_matrix, self.inv_global_motion, invert_matrix=True)
        add_pose_to_list(initial_extrinsic_matrix, self.global_extrinsic)

        # 2) We start building the posegraph (we don't add any edge since we have only one node [the curr frame])
        self.global_posegraph.add_node(initial_extrinsic_matrix)

        # 3) we perform integration
        # we load the rgbd
        curr_rgbd = RGBD(color_path=self.list_of_rgb[0], depth_path=self.list_of_depth[0], device=self.o3d_device)


        # integration part
        self.tsdf.build_3D_map(curr_rgbd.rgbd_tsdf, self.o3d_intrinsic, initial_extrinsic_matrix)
        #self.map.integrate(curr_rgbd, initial_extrinsic_matrix, 0)

        # we save the results
        #self.tsdf.save_pcd(self.pcd_save_path.replace("%", "0"))
        #self.tsdf.save_mesh(self.mesh_save_path.replace("%", "0"))
        #self.map.save_pcd(self.pcd_save_path.replace("%", "0"))
        ##self.map.save_mesh(self.mesh_save_path.replace("%", "0"))

        pcd = self.tsdf.extract_pcd()
        #pcd = self.map.scene_pcd.to_legacy()

        return curr_rgbd, pcd, initial_extrinsic_matrix

    def _sequential_loop(self, i):
        # we need to perform two operations
        # 1) pose estimation
        # 2) 3D reconstruction

        print(i)

        # load the rgbd
        print(self.list_of_rgb[i])
        curr_rgbd = RGBD(color_path=self.list_of_rgb[i], depth_path=self.list_of_depth[i], depth_scale=self.depth_scale, device=self.o3d_device)
        prev_rgbd = RGBD(color_path=self.list_of_rgb[i-1], depth_path=self.list_of_depth[i-1], depth_scale=self.depth_scale, device=self.o3d_device)

        # visual odometry
        transformation = self.vo.estimate_relative_pose_between(prev_frame=self.list_of_rgb[i-1], curr_frame=self.list_of_rgb[i], prev_rgbd=prev_rgbd, curr_rgbd=curr_rgbd, i=i)


        # get the absolute pose
        curr_absolute_pose = compute_curr_estimate_global_pose(self.global_extrinsic[-1], transformation)

        # store the computed matrices
        add_pose_to_list(transformation, self.global_motion)
        add_pose_to_list(transformation, self.inv_global_motion, invert_matrix=True)
        add_pose_to_list(curr_absolute_pose, self.global_extrinsic)

        # add them to the posegraph
        self.global_posegraph.add_node(curr_absolute_pose)
        self.global_posegraph.add_edge(transformation, i, i - 1, False)

        if i % self.num_posegraph_optim == 0:
            # step 1: we optimize the posegraph
            self.global_posegraph.optimize()

            # step 2: we update the poses
            prev_global_extr = self.global_extrinsic
            self.global_extrinsic = update_global_extrinsic(self.global_posegraph.pose_graph)

            if not np.array_equal(prev_global_extr, self.global_extrinsic):
                # this mean that the pose have been optimized so we are going also to update the map
                # we update the map

                # la mappa l'aggiorniamo ogni 100 frame
                pass

                self.tsdf = update_map_after_pg(self.global_extrinsic, self.list_of_rgb, self.list_of_depth, self.depth_scale, self.o3d_device, self.o3d_intrinsic)
            print(f"posegraph non fa nulla (?) -> {np.array_equal(prev_global_extr, self.global_extrinsic)}")
        else:
            # integration part
            print("Integrating ...")
            self.tsdf.build_3D_map(curr_rgbd.rgbd_tsdf, self.o3d_intrinsic, self.global_extrinsic[-1])
            #self.map.integrate(curr_rgbd, extrinsics=self.global_extrinsic[-1], id=i)

        # we save the results
        if i % 2000 == 0:
            self.tsdf = update_map_after_pg(self.global_extrinsic, self.list_of_rgb, self.list_of_depth, self.depth_scale,
                                            self.o3d_device, self.o3d_intrinsic)
        #self.tsdf.build_3D_map(curr_rgbd.rgbd_tsdf, self.o3d_intrinsic, self.global_extrinsic[-1])



        #self.tsdf.save_pcd(self.pcd_save_path.replace("%", str(i)))
        #self.tsdf.save_mesh(self.mesh_save_path.replace("%", str(i)))

        pcd = self.tsdf.extract_pcd()
        # pcd = self.map.extract_pcd()
        #pcd = self.map.scene_pcd.to_legacy()







        return curr_rgbd, prev_rgbd, pcd, self.global_extrinsic[-1]




if __name__ == "__main__":
    depth_map_path = "/home/gvide/Scrivania/slam_test/depth01"
    rgb_path = "/home/gvide/Scrivania/slam_test/image01"
    path_to_model = "/home/gvide/PycharmProjects/BodySLAM/MPEM/Model/9_best_model_gen_ab.pth"

    rgb_list = sorted(os.listdir(rgb_path))
    depth_list = sorted(os.listdir(depth_map_path))

    for i in range(len(rgb_list)):
        rgb_list[i] = os.path.join(rgb_path, rgb_list[i])
        depth_list[i] = os.path.join(depth_map_path, depth_list[i])

    slam = SLAM(rgb_list, depth_list, path_to_model)
    slam.main_loop_no_gui()
