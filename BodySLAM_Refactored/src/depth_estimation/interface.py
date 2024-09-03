"""
This file contains the main interface for depth estimation.
It defines the DepthEstimator class, which encapsulates the logic for
initializing depth estimation models and performing inference.
The class is designed to be flexible, allowing different model types
to be used for depth estimation.
"""

import warnings
import torch
from PIL import Image
from typing import Optional
import os


class DepthEstimator:
    '''A class to interface with ZOE for monocular depth estimation'''

    SUPPORTED_MODELS = ['ZoeD_N', 'ZoeD_K', 'ZoeD_NK']
    DEFAULT_MODEL = 'ZoeD_NK'

    def __init__(self, model_type: str = DEFAULT_MODEL):
        """
        Initialize the DepthEstimator with a specified model type.

        :param model_type: The model type to use for depth estimation
        """
        self.model = self._initialize_model(model_type)

    def _initialize_model(self, model_type: str) -> torch.nn.Module:
        """
        Initialize and return the specified depth estimation model.

        :param model_type: The type of model to initialize
        :return: Initialized PyTorch model
        """
        if model_type not in self.SUPPORTED_MODELS:
            warnings.warn(
                f"The model type '{model_type}' is not supported. Using default model '{self.DEFAULT_MODEL}'.")
            model_type = self.DEFAULT_MODEL

        # Fetch the latest MiDaS repo
        torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

        # Load the model from torch hub
        model = torch.hub.load("isl-org/ZoeDepth", model_type, pretrained=True)

        # Move model to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Model loaded on {device}")
        return model.to(device)

    def infer_depth_map(self, path_to_frame: str) -> Image.Image:
        """
        Infer the depth map from a monocular frame.

        :param path_to_frame: Path to the input frame
        :return: Depth map as a PIL Image
        """
        image = self.load_image(path_to_frame)
        return self.model.infer_pil(image, output_type="pil")

    @staticmethod
    def load_image(path: str) -> Image.Image:
        """
        Load an image from the given path and convert it to RGB.

        :param path: Path to the image file
        :return: PIL Image in RGB mode
        """
        image = Image.open(path)
        return image.convert('RGB')

    @staticmethod
    def save_depth_map(image: Image.Image, saving_path: str, extension: Optional[str] = None):
        """
        Save a depth map to the specified path.

        :param image: PIL Image of the depth map
        :param saving_path: Path to save the depth map
        :param extension: File extension for the saved image (optional)
        """
        if extension:
            saving_path = os.path.splitext(saving_path)[0] + '.' + extension.lstrip('.')

        image.save(saving_path)

    def debug(self, path_to_frame: str, saving_path: str):
        """
        Test all methods of this class for debugging purposes.

        :param path_to_frame: Path to the input image/frame
        :param saving_path: Path to the folder for saving output
        """
        tests = [
            ("load image", lambda: self.load_image(path_to_frame)),
            ("infer method", lambda: self.infer_depth_map(path_to_frame)),
            ("saving method", lambda: self.save_depth_map(Image.new('RGB', (100, 100)), saving_path))
        ]

        for test_name, test_func in tests:
            print(f"[DEBUG]: Testing {test_name}...")
            try:
                test_func()
                print(f"[DEBUG]: {test_name} status -> ok")
            except Exception as e:
                print(f"[DEBUG]: OPS :/ -> {e}")

