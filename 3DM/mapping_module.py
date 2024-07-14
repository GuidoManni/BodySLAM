'''
Add a description
'''
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy import stats
class MappingModule:
    def __init__(self, device, intrinsics_t, intrinsics, depth_scale):
        self.device = device
        self.intrinsics_t = intrinsics_t
        self.intrinsics = intrinsics
        self.depth_scale = depth_scale

        # Create an empty triangle mesh and an empty pcd to store the scene
        self.scene_mesh = o3d.t.geometry.TriangleMesh
        self.scene_pcd = o3d.t.geometry.PointCloud

        self.poisson_quantile = 0.01



    def reconstruction_settings(self, poisson_depth: int = 8, poisson_quantile: float = 0.01):
        self.poisson_depth = poisson_depth
        self.poisson_quantile = poisson_quantile

    def integrate(self, curr_rgbd, extrinsics, id):
        # step 1: check if it's the first integration
        if id == 0:
            # if it is empty then we are in the first iteration
            self._first_integration(curr_rgbd.rgbd_t, extrinsics)
        else:
            # step 2: compute the raycasting on the current scene with the new camera pose
            synthetic_dp = self._compute_synthetic_depth(extrinsics, id)

            # step 3: reconstruct synthetic pcd
            synthetic_pcd = o3d.geometry.PointCloud.create_from_depth_image(synthetic_dp, self.intrinsics, extrinsics)
            #o3d.visualization.draw_geometries([synthetic_pcd])

            # step 4: reconstruct curr pcd
            curr_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(curr_rgbd.rgbd, intrinsic=self.intrinsics, extrinsic=extrinsics)
            curr_pcd_t = o3d.t.geometry.PointCloud.create_from_rgbd_image(curr_rgbd.rgbd_t, intrinsics=self.intrinsics_t, extrinsics=extrinsics)

            # step 5: compute the distance between the synthetic pcd and the current pcd
            dists_synt_curr = np.array(synthetic_pcd.compute_point_cloud_distance(curr_pcd))

            # step 6: compute the distance between the curr pcd and the synthetic pcd
            dists_curr_synt = np.array(curr_pcd.compute_point_cloud_distance(synthetic_pcd))

            # step 7: get from dist_synt_curr the part to remove from the synthetic pcd
            ind_sc = np.where(dists_synt_curr < 0.01)[0]
            pcd_rm = synthetic_pcd.select_by_index(ind_sc)

            #o3d.visualization.draw_geometries([pcd_rm])

            # step 8: get from dist_curr_synt the part to integrate
            ind_cs = o3d.core.Tensor(np.where(dists_curr_synt > 0.01)[0]).to(self.device)
            pcd_add = curr_pcd_t.select_by_index(ind_cs)

            # step 9: now we use pcd_rm to remove the portion from the scene
            pcd_scene = self.scene_pcd.to_legacy()
            dist_scene_rm = np.array(pcd_scene.compute_point_cloud_distance(pcd_rm))
            ind_to_remove = np.where(dist_scene_rm > 0.01)[0]
            self.scene_pcd = o3d.t.geometry.PointCloud.from_legacy(pcd_scene.select_by_index(ind_to_remove)).to(self.device)
            #o3d.visualization.draw_geometries([self.scene_pcd.to_legacy()])
            #o3d.visualization.draw_geometries([pcd_scene])
            #self.scene_pcd += o3d.t.geometry.PointCloud.from_legacy(pcd_add)
            self.scene_pcd += pcd_add

            pcd_scene = self.scene_pcd.to_legacy()

            #pcd_scene = pcd_scene.voxel_down_sample(0.005)
            diameter = np.linalg.norm(
                np.asarray(pcd_scene.get_max_bound()) - np.asarray(pcd_scene.get_min_bound()))
            camera = o3d.core.Tensor([0, 0, diameter], o3d.core.float32).to(self.device)
            radius = diameter * 100

            _, pt_map = self.scene_pcd.hidden_point_removal(camera, radius)
            self.scene_pcd = self.scene_pcd.select_by_index(pt_map)

            #o3d.visualization.draw_geometries([pcd_scene])

            #self.scene_pcd = o3d.t.geometry.PointCloud.from_legacy(pcd_scene)

            if id % 1000 == 0:

                cl, ind = self.scene_pcd.remove_statistical_outliers(nb_neighbors=20,
                                                                 std_ratio=2.0)

                self.scene_pcd = cl

            self._update_scene_pcd()



            '''
            # step 2: compute the raycasting on the current scene with the new camera pose
            synthetic_dp = self._compute_synthetic_depth(extrinsics, id)

            # step 3: compute the residuals between the current real dp and the synthetic dp to get a mask
            residual_mask = self._compute_residual_mask(synthetic_dp, curr_rgbd.cv2_depth, id)

            # step 4: apply the mask to the real dp and its corresponding rgb
            rgbd_masked = self._rgbd_to_integrate(curr_rgbd, residual_mask)

            # apply the mask to the synthetic dp
            synthetic_dp_masked = o3d.t.geometry.Image(np.where(residual_mask == 1, synthetic_dp, np.nan))
            real_dp_masked = o3d.t.geometry.Image(np.where(residual_mask == 1, curr_rgbd.cv2_depth, np.nan))

            # step 5: reconstruct the pcds
            pcd_masked = o3d.t.geometry.PointCloud.create_from_depth_image(real_dp_masked, self.intrinsics, extrinsics)

            synthetic_pcd_masked = o3d.t.geometry.PointCloud.create_from_depth_image(depth=synthetic_dp_masked, intrinsics=self.intrinsics, extrinsics=extrinsics)

            # step 6: reconstruct the meshes
            mesh_masked = self._reconstruct_mesh_from_pcd(pcd_masked)
            synthetic_mesh_masked = self._reconstruct_mesh_from_pcd(synthetic_pcd_masked)
            #o3d.visualization.draw_geometries([synthetic_mesh_masked.to_legacy()])


            # step 7: integrate the meshes
            # First we remove the part that needs to be updated
            o3d.visualization.draw_geometries([self.scene_mesh.to_legacy(), synthetic_mesh_masked.to_legacy()])
            diff = self.scene_mesh.boolean_intersection(synthetic_mesh_masked)
            o3d.visualization.draw([diff])
            self.scene_mesh = self.scene_mesh.boolean_union(mesh_masked)
            o3d.visualization.draw_geometries([self.scene_mesh.to_legacy()])

            # step 8: we update also the pcd
            self._update_scene_pcd()
            self.scene_pcd.to_legacy()
            curr_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(curr_rgbd.rgbd_t, intrinsics=self.intrinsics_t, extrinsics=extrinsics)

            #o3d.visualization.draw_geometries([self.scene_pcd.to_legacy()])
            #o3d.visualization.draw_geometries([curr_pcd.to_legacy()])

            legacy_pcd_scene = self.scene_pcd.to_legacy()
            legacy_curr_pcd = curr_pcd.to_legacy()

            dists_1 = legacy_pcd_scene.compute_point_cloud_distance(legacy_curr_pcd)
            dists_1 = np.asarray(dists_1)
            print(np.max(dists_1))
            dists_2 = legacy_curr_pcd.compute_point_cloud_distance(legacy_pcd_scene)
            dists_2 = np.asarray(dists_2)
            print(np.max(dists_2))
            ind_to_remove = o3d.core.Tensor(np.where(dists_1 < np.quantile(dists_1, 0.01))[0]).to(self.device)
            ind_to_add = o3d.core.Tensor(np.where(dists_2 > np.quantile(dists_2, 0.5))[0]).to(self.device)
            self.scene_pcd = self.scene_pcd.select_by_index(ind_to_remove)
            curr_pcd = curr_pcd.select_by_index(ind_to_add)
            #o3d.visualization.draw_geometries([self.scene_pcd.to_legacy()])
            #o3d.visualization.draw_geometries([curr_pcd.to_legacy()])


            self.scene_pcd += curr_pcd

            self.scene_pcd = self.scene_pcd.voxel_down_sample(voxel_size=0.009)


            #self.scene_mesh = self._reconstruct_mesh_from_pcd(self.scene_pcd)

            #self._update_scene_pcd()

            #o3d.visualization.draw_geometries([self.scene_pcd.to_legacy()])

            # update the scene
            '''




    def _first_integration(self, curr_rgbd, extrinsics):
        # step 1: initialize the pcd with the first view
        self.scene_pcd = self.scene_pcd.create_from_rgbd_image(rgbd_image=curr_rgbd, intrinsics=self.intrinsics_t, extrinsics=extrinsics).to(self.device)

        # step 2: we reconstruct the mesh
        self.scene_mesh = self._reconstruct_mesh_from_pcd(self.scene_pcd)

    def _reconstruct_mesh_from_pcd(self, pcd, legacy = False, remove_low_support_vertices = True):
        # to reconstruct the mesh from a pcd we will use Poisson

        # step 1: pass the pcd to cpu and to legacy
        pcd = pcd.cpu().to_legacy()

        pcd.estimate_normals()


        pcd = pcd.voxel_down_sample(0.05)

        #o3d.visualization.draw_geometries([pcd])

        # step 2: use poisson to reconstruct the mesh
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=self.poisson_depth)

        if remove_low_support_vertices:
            vertices_to_remove = densities < np.quantile(densities, self.poisson_quantile)
            mesh.remove_vertices_by_mask(vertices_to_remove)

        if legacy:
            return mesh
        else:
            return o3d.t.geometry.TriangleMesh.from_legacy(mesh)


    def _compute_synthetic_depth(self, camera_pose, i):
        # step 1: create the scene
        raycasting_scene = self._create_raycasting_scene()

        # convert the camera pose to a tensor
        camera_pose_t = o3d.core.Tensor(camera_pose)

        # step 2: create raycast object
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(self.intrinsics_t, camera_pose_t, width_px=600, height_px=480)

        # step 3: perform raycast
        raycast_result = raycasting_scene.cast_rays(rays)

        # get a depth map from the resulting raycast
        dp = raycast_result['t_hit'].numpy()

        dp = self._preprocess_dp_for_visualization(dp)

        # save the dp
        save_path_img = "/home/gvide/Scrivania/slam_test/synthetic_dp/" + str(i) + ".png"
        save_path_np = save_path_img.replace(".png", "")
        plt.imsave(save_path_img, dp, cmap='gray')  # Use cmap='gray' for grayscale images
        np.save(save_path_np, dp)

        return o3d.geometry.Image(dp)

    def _create_raycasting_scene(self):
        # step 2: create scene
        scene = o3d.t.geometry.RaycastingScene()

        # step 3: add the mesh to the scene
        scene.add_triangles(self.scene_mesh.cpu())

        return scene

    def _preprocess_dp_for_visualization(self, depth_map: np.ndarray) -> np.ndarray:
        # Replacing infinite values with the maximum finite value
        finite_max = np.nanmax(depth_map[np.isfinite(depth_map)])
        depth_map_processed = np.where(np.isfinite(depth_map), depth_map, finite_max)

        # Rescaling the depth map to range [0, 1]
        depth_map_rescaled = (depth_map_processed - depth_map_processed.min()) / (
                depth_map_processed.max() - depth_map_processed.min())

        return depth_map_rescaled

    def _compute_residual_mask(self, synthetic_dp, real_dp, i):
        # convert the real_dp to a float32
        real_dp = real_dp.astype('float32')

        # compute the residuals
        residuals = synthetic_dp - real_dp

        # Calculate the median of the residuals
        median_residuals = np.median(residuals)

        # Compute the Median Absolute Deviation (MAD)
        mad = stats.median_abs_deviation(residuals)

        # Set a threshold based on a multiple of the MAD, commonly 2.5 or 3 times the MAD is used
        mad_threshold = median_residuals + 1.5 * mad

        # Create the mask: 1 for residuals greater than the threshold (big errors), 0 otherwise
        mask_big_errors = np.where(residuals > mad_threshold, 1, 0)

        save_path_img = "/home/gvide/Scrivania/slam_test/residual_mask/" + str(i) + ".png"

        plt.imsave(save_path_img, mask_big_errors, cmap='gray')  # Use cmap='gray' for grayscale images

        return mask_big_errors


    def _rgbd_to_integrate(self, rgbd, residual_mask):
        residual_mask_rgb = residual_mask[:, :, np.newaxis]
        real_depth_masked = np.where(residual_mask == 1, rgbd.cv2_depth, np.nan)
        real_rgb_masked = np.where(residual_mask_rgb == 1, rgbd.cv2_color, np.nan)
        color = o3d.t.geometry.Image(real_rgb_masked)
        depth = o3d.t.geometry.Image(real_depth_masked.astype(np.float32))
        rgbd_masked = o3d.t.geometry.RGBDImage(color, depth)

        return rgbd_masked

    def _update_scene_pcd(self):
        mesh = self.scene_mesh.to_legacy()

        scene_pcd = mesh.sample_points_uniformly(number_of_points=10)
        self.scene_pcd.from_legacy(scene_pcd)