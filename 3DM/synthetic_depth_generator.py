'''
Add a description
'''

import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

from scipy import stats


def preprocess_dp_for_visualization(depth_map: np.ndarray) -> np.ndarray:
    # Replacing infinite values with the maximum finite value
    finite_max = np.nanmax(depth_map[np.isfinite(depth_map)])
    depth_map_processed = np.where(np.isfinite(depth_map), depth_map, finite_max)

    # Rescaling the depth map to range [0, 1]
    depth_map_rescaled = (depth_map_processed - depth_map_processed.min()) / (
                depth_map_processed.max() - depth_map_processed.min())

    return depth_map_rescaled

def create_scene(mesh):
    # convert the mesh to tensor mesh
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # step 2: create scene
    scene = o3d.t.geometry.RaycastingScene()

    # step 3: add the mesh to the scene
    mesh_id = scene.add_triangles(mesh)
    print(mesh_id)

    return scene


def _get_depth_map_from_rc(rays, raycast_result, extrinsic_t, intrinsic_t, depth_width, depth_height):
    # extraction of the distance to the intersection between the ray and the mesh (inf mean non intersection)
    hit = raycast_result['t_hit'].isfinite()

    # exctract the points
    points = rays[hit][:,:3] + rays[hit][:,3:]*raycast_result['t_hit'][hit].reshape((-1,1))

    # transfor the points to a pcd
    pcd_t = o3d.t.geometry.PointCloud(points)
    pcd = pcd_t.to_legacy()
    points = np.asarray(pcd.points)

    resolution = [depth_height, depth_width]

    # Normalize x, y to grid coordinates, and depth (z) to the desired range
    x = np.interp(points[:, 0], (points[:, 0].min(), points[:, 0].max()), (0, resolution[1] - 1)).astype(int)
    y = np.interp(points[:, 1], (points[:, 1].min(), points[:, 1].max()), (0, resolution[0] - 1)).astype(int)
    z = np.interp(points[:, 2], (points[:, 2].min(), points[:, 2].max()), (0, 65535)).astype(np.uint16)

    # Initialize depth map
    depth_map = np.zeros((resolution[0], resolution[1]), dtype=np.uint16)

    depth_map[y, x] = z



    #o3d.visualization.draw_geometries([pcd.to_legacy()])

    # from pcd to depth map
    depth_o3d = pcd_t.project_to_depth_image(depth_width, depth_height, intrinsic_t, extrinsic_t)

    #depth_map = np.asarray(depth_map)

    return depth_map, depth_o3d


def compute_synthetic_depth(mesh, intrinsic, camera_pose, i):
    # step 1: create the scene
    scene = create_scene(mesh)

    camera_pose_t = o3d.core.Tensor(camera_pose)

    # step 2: create raycast object
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(intrinsic, camera_pose_t, width_px=600, height_px=480)

    # step 3: perform raycast
    raycast_result = scene.cast_rays(rays)

    # get a depth map from the resulting raycast
    dp = raycast_result['t_hit'].numpy()

    dp = preprocess_dp_for_visualization(dp)

    # save the dp
    save_path_img = "/home/gvide/Scrivania/slam_test/synthetic_dp/" + str(i) + ".png"
    save_path_np = save_path_img.replace(".png", "")
    plt.imsave(save_path_img, dp, cmap='gray')  # Use cmap='gray' for grayscale images
    np.save(save_path_np, dp)

    return dp

def compute_residuals_between(synthetic_dp, real_dp, i):
    # convert the real_dp to a float32
    real_dp = real_dp.astype('float32')

    # compute the residuals
    residuals = synthetic_dp - real_dp

    mean_residuals = np.mean(residuals)
    std_residuals = np.std(residuals)

    # Calculate the median of the residuals
    median_residuals = np.median(residuals)

    # Compute the Median Absolute Deviation (MAD)
    mad = stats.median_abs_deviation(residuals)

    # Set a threshold based on a multiple of the MAD, commonly 2.5 or 3 times the MAD is used
    mad_threshold = median_residuals + 1.5 * mad

    #big_error_threshold = mean_residuals + 1.5 * std_residuals
    #threshold = threshold_otsu(residuals)

    # Calculate the maximum absolute residual to set the fixed percentage threshold
    #max_absolute_residual = np.max(np.abs(residuals))

    # Set the threshold to 10% of the maximum absolute residual
    #fixed_percentage_threshold = 0.5 * max_absolute_residual

    # Create the mask: 1 for residuals greater than the threshold (big errors), 0 otherwise
    mask_big_errors = np.where(residuals > mad_threshold, 1, 0)

    save_path_img = "/home/gvide/Scrivania/slam_test/residual_mask/" + str(i) + ".png"

    plt.imsave(save_path_img, mask_big_errors, cmap='gray')  # Use cmap='gray' for grayscale images

    return mask_big_errors



