import unittest
import os
from PIL import Image
from src.depth_estimation.interface import DepthEstimator


class TestDepthEstimator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Go up two directories to reach the project root
        cls.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

        # Define paths
        cls.test_image_path = os.path.join(cls.project_root, 'tests', 'resources', 'depth_estimation',
                                           'input_image.jpg')
        cls.test_output_path = os.path.join(cls.project_root, 'tests', 'resources', 'depth_estimation',
                                            'output_depth_map.png')

        # Ensure output directory exists
        os.makedirs(os.path.dirname(cls.test_output_path), exist_ok=True)

    def test_depth_estimation_pipeline(self):
        # 1. Load the model
        estimator = DepthEstimator()
        self.assertIsNotNone(estimator.model, "Model should be loaded")

        # 2. Load the image
        image = DepthEstimator.load_image(self.test_image_path)
        self.assertIsInstance(image, Image.Image, "Should load a PIL Image")
        self.assertEqual(image.mode, 'RGB', "Image should be in RGB mode")

        # 3. Compute the depth map of the image
        depth_map = estimator.infer_depth_map(self.test_image_path)
        self.assertIsInstance(depth_map, Image.Image, "Should return a PIL Image")

        # 4. Save the computed depth map
        DepthEstimator.save_depth_map(depth_map, self.test_output_path)
        self.assertTrue(os.path.exists(self.test_output_path), "Depth map should be saved")

        # Optional: Check the saved image
        saved_image = Image.open(self.test_output_path)
        self.assertIsInstance(saved_image, Image.Image, "Should be able to open the saved depth map")


if __name__ == '__main__':
    unittest.main()