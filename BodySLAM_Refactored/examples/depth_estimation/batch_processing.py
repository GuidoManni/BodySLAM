"""
Example of batch processing multiple images for depth estimation and colorization.
"""

import os
import numpy as np
import torch
import matplotlib
from PIL import Image
from src.depth_estimation.interface import DepthEstimator

def colorize(value, vmin=None, vmax=None, cmap='viridis', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
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

def process_image(estimator, input_path, output_path, colormap='viridis', invalid_val=0):
    """Process a single image: estimate depth, colorize, and save."""
    # Compute depth map
    depth_map = estimator.infer_depth_map(input_path)
    depth_array = np.array(depth_map)

    # Colorize depth map
    colorized_depth = colorize(depth_array, cmap=colormap, invalid_val=invalid_val)

    # Save colorized depth map
    Image.fromarray(colorized_depth).save(output_path)
    print(f"Processed and saved: {output_path}")

def process_images(input_dir, output_dir, colormap='viridis', invalid_val=0):
    """Process all images in the input directory and save results to the output directory."""
    estimator = DepthEstimator()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"depth_{os.path.splitext(filename)[0]}.png")

            process_image(estimator, input_path, output_path, colormap, invalid_val)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'resources', 'input_images')
    output_dir = os.path.join(current_dir, 'resources', 'output')

    print(f"Processing images from: {input_dir}")
    print(f"Saving results to: {output_dir}")

    process_images(input_dir, output_dir)

    print("Batch processing complete!")

if __name__ == "__main__":
    main()