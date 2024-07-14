from typing import Tuple

import PIL.Image
import numpy as np
import open3d as o3d
import torch
import cv2
from PIL import Image

from tsdf import TSDF

def check_o3d_device() -> o3d.core.Device:
    '''
    Check if gpu is available in the system
    :return: device (cpu or gpu)
    '''
    cuda_supported = o3d.core.cuda.is_available()
    device = o3d.core.Device("CUDA:0") if cuda_supported else o3d.core.Device("CPU:0")
    return device

def check_torch_device() -> torch.device:
    '''
    Check if gpu si available in the system
    :return: device (cpu or gpu)
    '''
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    return device
def device_handler() -> Tuple[o3d.core.Device, torch.device]:
    '''
    check if gpu is available in the system
    :return: device (cpu or gpu)
    '''
    o3d_device = check_o3d_device()
    torch_device = check_torch_device()

    return o3d_device, torch_device

def is_np_ndarray(matrix: np.ndarray) -> bool:
    '''
    Will check if the matrix is a numpy array
    :param matrix:
    :return:
    '''
    if isinstance(matrix, np.ndarray):
        return True

def get_o3d_intrinsic(frame_width: int, frame_height: int, fx: float, fy: float, cx: float, cy: float)-> Tuple[o3d.camera.PinholeCameraIntrinsic, o3d.core.Tensor]:
    '''
    This function build the intrinsic matrix used by open3d
    :param frame_width:
    :param frame_height:
    :param fx:
    :param fy:
    :param cx:
    :param cy:
    :return: intrinsic matrix
    '''
    # initialize o3d intrinsic with prime-sense intrinsic
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

    # set the new values passed
    o3d_intrinsic.set_intrinsics(width=frame_width, height=frame_height, fx=fx, fy=fy, cx=cx, cy=cy)

    # create also a tensor version for gpu element
    o3d_intrinsic_t = o3d.core.Tensor(o3d_intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)

    return o3d_intrinsic, o3d_intrinsic_t


def add_pose_to_list(matrix: np.ndarray, pose_list: list[np.ndarray], invert_matrix: bool = False):
    '''
    Update the provided pose list with the latest pose
    :param matrix: transformation/absolute pose [4x4 matrix]
    :param pose_list: list of poses
    :param invert_matrix: if true will invert the matrix
    :return:
    '''

    if not is_np_ndarray(matrix):
        # if the passed matrix is not a numpy array then we will convert it
        matrix = matrix.cpu().numpy()
    if invert_matrix:
        matrix = np.linalg.inv(matrix)
    pose_list.append(matrix)


def update_global_extrinsic(global_pose_graph):
    global_extrinsic = [node.pose for node in global_pose_graph.nodes]
    return global_extrinsic


def ensure_so3_v2(matrix: np.ndarray) -> np.ndarray:
    """
    Projects a 3x3 matrix to the closest SO(3) matrix using an alternative method.
    """

    U, _, Vt = np.linalg.svd(matrix)


    # Creating the intermediate diagonal matrix
    D = np.eye(3)
    D[2, 2] = np.linalg.det(U) * np.linalg.det(Vt)

    # Compute the closest rotation matrix
    R = np.dot(U, np.dot(D, Vt))

    return R

def compute_curr_estimate_global_pose(global_extrinsic: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    '''
    Compute the global current pose from the motion matrix/transformation obtained from visual odometry
    :param global_extrinsic_list: list of global poses
    :param transformation: 4x4 matrix describing the relative motion between two frames
    :return: curr_global_pose: current global pose
    '''

    curr_global_pose = np.dot(global_extrinsic, transformation)
    # ensure that the obtained pose is so3 valid
    curr_global_pose[:3, :3] = ensure_so3_v2(curr_global_pose[:3, :3])

    return curr_global_pose

def update_map_after_pg(global_extrinsic, list_of_rgb, list_of_depth, depth_scale, device, intrinsic):
    tsdf = TSDF()
    #map = MAP()
    n_frames_processed = len(global_extrinsic)

    for i in range(n_frames_processed):
        rgbd = RGBD(color_path=list_of_rgb[i], depth_path=list_of_depth[i], depth_scale=depth_scale, device=device)

        tsdf.build_3D_map(rgbd.rgbd_tsdf, intrinsic, global_extrinsic[i])
        #map.integrate(rgbd, global_extrinsic[i])

    return tsdf


def estimate_similarity_transformation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transformation (rotation, scale, translation) from source to target (such as the Sim3 group).
    """
    k, n = source.shape

    mx = source.mean(axis=1)
    my = target.mean(axis=1)
    source_centered = source - np.tile(mx, (n, 1)).T
    target_centered = target - np.tile(my, (n, 1)).T

    sx = np.mean(np.sum(source_centered ** 2, axis=0))
    sy = np.mean(np.sum(target_centered ** 2, axis=0))

    Sxy = (target_centered @ source_centered.T) / n

    U, D, Vt = np.linalg.svd(Sxy, full_matrices=True, compute_uv=True)
    V = Vt.T
    rank = np.linalg.matrix_rank(Sxy)
    #if rank < k:
    #    raise ValueError("Failed to estimate similarity transformation")

    S = np.eye(k)
    if np.linalg.det(Sxy) < 0:
        S[k - 1, k - 1] = -1

    R = U @ S @ V.T

    s = np.trace(np.diag(D) @ S) / sx
    t = my - s * (R @ mx)

    return R, s, t


class RGBD:
    def __init__(self, color_path: str, depth_path: str, device: o3d.core.Device, depth_scale: int = 1000, depth_trunc: float = 3.0):
        self.color_path = color_path
        self.depth_path = depth_path
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc
        self.device = device

        # store legacy color/depth/rgbd
        self.o3d_color = self._read_img(color_path)
        self.o3d_depth = self._read_img(depth_path)
        self.rgbd = self._read_rgbd()

        # store rgbd for tsdf integration
        self.rgbd_tsdf = self._read_rgbd_for_tsdf()

        # store tensor color/depth/rgbd
        self.o3d_t_color = self._read_t_img(color_path).to(device)
        self.o3d_t_depth = self._read_t_img(depth_path).to(device)
        self.rgbd_t = self._read_t_rgbd().to(device)

        # store cv2/np color/depth
        self.cv2_color = self._read_color_cv2(color_path)
        self.cv2_depth = self._read_depth_cv2(depth_path)

        # store colored_depth
        self.colored_depth = self._compute_colored_depth(depth_path)

        # store pil color
        self.pil_color = self._read_color_PIL(color_path)

        # compute min/max used for 3D reconstruction
        self.depth_min, self.depth_max = self._compute_min_max_depth()

        # compute width/height
        self.height, self.width = self._compute_img_shape(self.cv2_depth)

    def _read_img(self, img_path: str) -> o3d.geometry.Image:
        return o3d.io.read_image(img_path)

    def _read_rgbd(self) -> o3d.geometry.RGBDImage:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(self.o3d_color, self.o3d_depth, depth_scale=self.depth_scale, depth_trunc=self.depth_trunc)

        return rgbd
    def _read_rgbd_for_tsdf(self) -> o3d.geometry.RGBDImage:
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(self.o3d_color, self.o3d_depth, depth_scale=self.depth_scale,
                                                                  depth_trunc=self.depth_trunc, convert_rgb_to_intensity = False)

        return rgbd
    def _read_t_img(self, img_path: str) -> o3d.t.geometry.Image:
        return o3d.t.io.read_image(img_path)

    def _read_t_rgbd(self) -> o3d.t.geometry.RGBDImage:
        rgbd_t = o3d.t.geometry.RGBDImage(self.o3d_t_color, self.o3d_t_depth)
        return rgbd_t

    def _read_color_cv2(self, img_path: str) -> np.ndarray:
        return cv2.imread(img_path)

    def _read_depth_cv2(self, depth_path: str) -> np.ndarray:
        return cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / self.depth_scale
        #return cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)

    def _read_color_PIL(self, img_path: str) -> PIL.Image.Image:
        return Image.open(img_path)

    def _compute_min_max_depth(self) -> Tuple[float, float]:
        min = np.amin(self.cv2_depth)
        max = np.amax(self.cv2_depth)

        return min, max

    def _compute_img_shape(self, img: np.ndarray) -> Tuple[int, int]:
        height = img.shape[0]
        width = img.shape[1]

        return height, width

    def _compute_colored_depth(self, depth_path: str) -> np.ndarray:
        # Load your 16-bit depth image
        # Replace 'path_to_your_depth_image.png' with your actual image path
        depth_image = cv2.imread(depth_path, -1)  # -1 ensures the image is read in its original depth

        # Normalize the depth image to 8-bit for visualization
        # Find the maximum and minimum values in the depth image
        min_val, max_val, _, _ = cv2.minMaxLoc(depth_image)
        # Scale the values to lie between 0 and 255
        normalized_depth = np.uint8(255 * (depth_image - min_val) / (max_val - min_val))

        # Apply a colormap for better visualization
        colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

        return o3d.geometry.Image(colored_depth)