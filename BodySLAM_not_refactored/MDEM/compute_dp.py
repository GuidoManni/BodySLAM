'''
Add a description
'''
import os

from mdem_interface import MDEMInterface
mdem_interface = MDEMInterface()
def compute_dp(path_to_rgb: str, output_path_for_dp: str):
    content_path = sorted(os.listdir(path_to_rgb))

    for content in content_path:
        print(f"[INFO]: computing dp for {content}")
        dp_path = os.path.join(output_path_for_dp, content.replace(".jpg", ".png"))
        rgb_path = os.path.join(path_to_rgb, content)

        dp = mdem_interface.infer_monocular_depth_map(rgb_path)

        mdem_interface.save_depth_map(dp, dp_path)

input_path = "/home/gvide/Scrivania/BodySLAM Results/3DM/BodySLAM/highcam_small_intestine_trajectory_1/Frames"

output_path = "/home/gvide/Scrivania/BodySLAM Results/3DM/BodySLAM/highcam_small_intestine_trajectory_1/depth"

compute_dp(input_path, output_path)