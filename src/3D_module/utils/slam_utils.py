from typing import Tuple
import PIL.Image
import numpy as np
import open3d as o3d
import torch
import cv2
from PIL import Image
from .tsdf import TSDF

def check_o3d_device() -> o3d.core.Device:
    """
    Check if GPU is available in the system.
    :return: device (CPU or GPU)
    """
    cuda_supported = o3d.core.cuda.is_available()
    return o3d.core.Device("CUDA:0") if cuda_supported else o3d.core.Device("CPU:0")

def check_torch_device() -> torch.device:
    """
    Check if GPU is available in the system.
    :return: device (CPU or GPU)
    """
    cuda_available = torch.cuda.is_available()
    return torch.device("cuda" if cuda_available else "cpu")

def device_handler() -> Tuple[o3d.core.Device, torch.device]:
    """
    Check if GPU is available in the system.
    :return: device (CPU or GPU)
    """
    return check_o3d_device(), check_torch_device()

def is_np_ndarray(matrix: np.ndarray) -> bool:
    """
    Check if the matrix is a numpy array.
    :param matrix: input matrix
    :return: True if matrix is a numpy array, False otherwise
    """
    return isinstance(matrix, np.ndarray)

def get_o3d_intrinsic(frame_width: int, frame_height: int, fx: float, fy: float, cx: float, cy: float) -> Tuple[o3d.camera.PinholeCameraIntrinsic, o3d.core.Tensor]:
    """
    Build the intrinsic matrix used by Open3D.
    :param frame_width: frame width
    :param frame_height: frame height
    :param fx: focal length x
    :param fy: focal length y
    :param cx: principal point x
    :param cy: principal point y
    :return: intrinsic matrix
    """
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    o3d_intrinsic.set_intrinsics(width=frame_width, height=frame_height, fx=fx, fy=fy, cx=cx, cy=cy)
    o3d_intrinsic_t = o3d.core.Tensor(o3d_intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)
    return o3d_intrinsic, o3d_intrinsic_t

def add_pose_to_list(matrix: np.ndarray, pose_list: list[np.ndarray], invert_matrix: bool = False):
    """
    Update the provided pose list with the latest pose.
    :param matrix: transformation/absolute pose [4x4 matrix]
    :param pose_list: list of poses
    :param invert_matrix: if true will invert the matrix
    """
    if not is_np_ndarray(matrix):
        matrix = matrix.cpu().numpy()
    if invert_matrix:
        matrix = np.linalg.inv(matrix)
    pose_list.append(matrix)

def update_global_extrinsic(global_pose_graph):
    """
    Update the global extrinsic list from the pose graph.
    :param global_pose_graph: pose graph
    :return: updated global extrinsic list
    """
    return [node.pose for node in global_pose_graph.nodes]

def ensure_so3_v2(matrix: np.ndarray) -> np.ndarray:
    """
    Projects a 3x3 matrix to the closest SO(3) matrix using an alternative method.
    :param matrix: input matrix
    :return: closest SO(3) matrix
    """
    U, _, Vt = np.linalg.svd(matrix)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(U) * np.linalg.det(Vt)
    return np.dot(U, np.dot(D, Vt))

def compute_curr_estimate_global_pose(global_extrinsic: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """
    Compute the global current pose from the motion matrix/transformation obtained from visual odometry.
    :param global_extrinsic: list of global poses
    :param transformation: 4x4 matrix describing the relative motion between two frames
    :return: current global pose
    """
    curr_global_pose = np.dot(global_extrinsic, transformation)
    curr_global_pose[:3, :3] = ensure_so3_v2(curr_global_pose[:3, :3])
    return curr_global_pose

def update_map_after_pg(global_extrinsic, list_of_rgb, list_of_depth, depth_scale, device, intrinsic):
    """
    Update the map after pose graph optimization.
    :param global_extrinsic: list of global poses
    :param list_of_rgb: list of RGB images
    :param list_of_depth: list of depth images
    :param depth_scale: depth scale
    :param device: device (CPU or GPU)
    :param intrinsic: camera intrinsic parameters
    :return: updated TSDF map
    """
    tsdf = TSDF()
    n_frames_processed = len(global_extrinsic)
    for i in range(n_frames_processed):
        rgbd = RGBD(color_path=list_of_rgb[i], depth_path=list_of_depth[i], depth_scale=depth_scale, device=device)
        tsdf.build_3D_map(rgbd.rgbd_tsdf, intrinsic, global_extrinsic[i])
    return tsdf

def estimate_similarity_transformation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Estimate similarity transformation (rotation, scale, translation) from source to target (such as the Sim3 group).
    :param source: source points
    :param target: target points
    :return: similarity transformation (R, s, t)
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
        self._initialize_images()

    def _initialize_images(self):
        self.o3d_color = self._read_img(self.color_path)
        self.o3d_depth = self._read_img(self.depth_path)
        self.rgbd = self._read_rgbd()
        self.rgbd_tsdf = self._read_rgbd_for_tsdf()
        self.o3d_t_color = self._read_t_img(self.color_path).to(self.device)
        self.o3d_t_depth = self._read_t_img(self.depth_path).to(self.device)
        self.rgbd_t = self._read_t_rgbd().to(self.device)
        self.cv2_color = self._read_color_cv2(self.color_path)
        self.cv2_depth = self._read_depth_cv2(self.depth_path)
        self.colored_depth = self._compute_colored_depth(self.depth_path)
        self.pil_color = self._read_color_PIL(self.color_path)
        self.depth_min, self.depth_max = self._compute_min_max_depth()
        self.height, self.width = self._compute_img_shape(self.cv2_depth)

    def _read_img(self, img_path: str) -> o3d.geometry.Image:
        return o3d.io.read_image(img_path)

    def _read_rgbd(self) -> o3d.geometry.RGBDImage:
        return o3d.geometry.RGBDImage.create_from_color_and_depth(self.o3d_color, self.o3d_depth, depth_scale=self.depth_scale, depth_trunc=self.depth_trunc)

    def _read_rgbd_for_tsdf(self) -> o3d.geometry.RGBDImage:
        return o3d.geometry.RGBDImage.create_from_color_and_depth(self.o3d_color, self.o3d_depth, depth_scale=self.depth_scale, depth_trunc=self.depth_trunc, convert_rgb_to_intensity=False)

    def _read_t_img(self, img_path: str) -> o3d.t.geometry.Image:
        return o3d.t.io.read_image(img_path)

    def _read_t_rgbd(self) -> o3d.t.geometry.RGBDImage:
        return o3d.t.geometry.RGBDImage(self.o3d_t_color, self.o3d_t_depth)

    def _read_color_cv2(self, img_path: str) -> np.ndarray:
        return cv2.imread(img_path)

    def _read_depth_cv2(self, depth_path: str) -> np.ndarray:
        return cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / self.depth_scale

    def _read_color_PIL(self, img_path: str) -> PIL.Image.Image:
        return Image.open(img_path)

    def _compute_min_max_depth(self) -> Tuple[float, float]:
        return np.amin(self.cv2_depth), np.amax(self.cv2_depth)

    def _compute_img_shape(self, img: np.ndarray) -> Tuple[int, int]:
        return img.shape[0], img.shape[1]

    def _compute_colored_depth(self, depth_path: str) -> np.ndarray:
        depth_image = cv2.imread(depth_path, -1)
        min_val, max_val, _, _ = cv2.minMaxLoc(depth_image)
        normalized_depth = np.uint8(255 * (depth_image - min_val) / (max_val - min_val))
        return o3d.geometry.Image(cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET))