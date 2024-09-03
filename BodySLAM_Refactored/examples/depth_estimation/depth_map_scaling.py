"""
Example of scaling and colorizing a depth map for better visualization.
"""

import os
import numpy as np
import torch
import matplotlib
from PIL import Image
from src.depth_estimation.interface import DepthEstimator

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask], 2) if vmin is None else vmin
    vmax = np.percentile(value[mask], 85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        value = value * 0.

    # grey out the invalid values
    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)

    value = cmapper(value, bytes=True)  # (nxmx4)
    img = value[...]
    img[invalid_mask] = background_color

    if gamma_corrected:
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)

    return img

def print_depth_info(depth_array):
    """Print diagnostic information about the depth array."""
    print(f"Depth array shape: {depth_array.shape}")
    print(f"Depth array dtype: {depth_array.dtype}")
    print(f"Min depth: {depth_array.min()}")
    print(f"Max depth: {depth_array.max()}")
    print(f"Mean depth: {depth_array.mean()}")

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_image = os.path.join(current_dir, 'resources', 'input_images', 'image1.jpg')
    output_image_color = os.path.join(current_dir, 'resources', 'output', 'colorized_depth_map1.png')

    # Load the model and compute depth map
    estimator = DepthEstimator()
    depth_map = estimator.infer_depth_map(input_image)
    print("Depth map computed")

    # Print depth map info
    depth_array = np.array(depth_map)
    print_depth_info(depth_array)

    # Colorize the depth map
    colorized_depth = colorize(depth_array, cmap='viridis', invalid_val=0)
    print("Depth map colorized")

    # Save the colorized depth map
    Image.fromarray(colorized_depth).save(output_image_color)
    print(f"Colorized depth map saved: {output_image_color}")

if __name__ == "__main__":
    main()