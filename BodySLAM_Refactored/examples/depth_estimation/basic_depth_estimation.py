"""
Example of a basic depth estimation pipeline using the DepthEstimator class.
"""

import os
from src.depth_estimation.interface import DepthEstimator

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_image = os.path.join(current_dir, 'resources', 'input_images', 'image1.jpg')
    output_image = os.path.join(current_dir, 'resources', 'output', 'depth_map1.png')

    # 1. Load the model
    estimator = DepthEstimator()

    # 2. Load the image
    image = DepthEstimator.load_image(input_image)
    print(f"Loaded image: {input_image}")

    # 3. Compute the depth map
    depth_map = estimator.infer_depth_map(input_image)
    print("Depth map computed")

    # 4. Save the depth map
    DepthEstimator.save_depth_map(depth_map, output_image)
    print(f"Depth map saved: {output_image}")

if __name__ == "__main__":
    main()