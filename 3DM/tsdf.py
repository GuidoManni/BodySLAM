import open3d as o3d
import numpy as np
from copy import deepcopy

class TSDF:
    def __init__(self, voxel_length: float = 0.006, sdf_trunc: float = 0.2):
        self.tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
                    voxel_length=voxel_length,
                    sdf_trunc=sdf_trunc,
                    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                    volume_unit_resolution = 16,
                    depth_sampling_stride = 8)

    def build_3D_map(self, rgbd: o3d.geometry.RGBDImage, intrinsic: o3d.camera.PinholeCameraIntrinsic, extrinsic):
        '''
        This function reconstruct the 3D model from the pseudo-rgbd using TSDF
        :param rgbd: pseudo-rgbd
        :param intrinsic: intrinsic parameter of the camera
        :param extrinsic: the global position of the camera
        :return:
        '''
        self.tsdf.integrate(rgbd, intrinsic, extrinsic)
    def build_copy_3D_map(self, rgbd: o3d.geometry.RGBDImage, intrinsic: o3d.camera.PinholeCameraIntrinsic, extrinsic):
        tsdf_copy = deepcopy(self.tsdf)

        tsdf_copy.integrate(rgbd, intrinsic, extrinsic)

        return tsdf_copy

    def save_pcd(self, saving_path: str):
        '''
        Save the generated pointcloud
        :param saving_path: path to file
        :return:
        '''
        pcd = self.tsdf.extract_point_cloud()
        o3d.io.write_point_cloud(saving_path, pcd)

    def extract_pcd(self):
        return self.tsdf.extract_point_cloud()

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        return self.tsdf.extract_triangle_mesh()

    def save_mesh(self, saving_path: str):
        '''
        Save the generated mesh
        :param saving_path: path to file
        :return:
        '''
        mesh = self.extract_mesh()
        o3d.io.write_triangle_mesh(saving_path, mesh)



class MAP:
    def __init__(self, width, height, intrinsic, device, depth_scale, voxel_size = 0.0058, block_count = 40000, trunc_voxel_multiplier = 8.0):
        T_frame_to_model = o3d.core.Tensor(np.identity(4))
        self.model = o3d.t.pipelines.slam.Model(voxel_size, 16,
                                           block_count, T_frame_to_model,
                                           device)
        self.input_frame = o3d.t.pipelines.slam.Frame(width, height,
                                                 intrinsic, device)

        self.raycast_frame = o3d.t.pipelines.slam.Frame(width,
                                                   height, intrinsic,
                                                   device)

        self.depth_scale = depth_scale
        self.trunc_voxel_multiplier = trunc_voxel_multiplier
    def integrate(self, curr_rgbd, i, curr_global_pose):
        self.input_frame.set_data_from_image('depth', curr_rgbd.o3d_t_depth)
        self.input_frame.set_data_from_image('color', curr_rgbd.o3d_t_color)

        self.model.update_frame_pose(i, o3d.core.Tensor(curr_global_pose))

        depth_max = curr_rgbd.depth_max
        depth_min = curr_rgbd.depth_min

        self.model.integrate(self.input_frame, self.depth_scale, depth_max, self.trunc_voxel_multiplier)
        self.model.synthesize_model_frame(self.raycast_frame, self.depth_scale,
                                     depth_min, depth_max,
                                     self.trunc_voxel_multiplier, False)


    def extract_pcd(self):
        return self.model.voxel_grid.extract_point_cloud().cpu().to_legacy()

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        return self.model.voxel_grid.extract_triangle_mesh().cpu().to_legacy()

    def save_pcd(self, saving_path: str):
        '''
        Save the generated pointcloud
        :param saving_path: path to file
        :return:
        '''
        pcd = self.extract_pcd()
        o3d.io.write_point_cloud(saving_path, pcd)

    def save_mesh(self, saving_path: str):
        '''
        Save the generated mesh
        :param saving_path: path to file
        :return:
        '''
        mesh = self.extract_mesh()
        o3d.io.write_triangle_mesh(saving_path, mesh)