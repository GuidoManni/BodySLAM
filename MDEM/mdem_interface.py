'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 30/08/23


Description:
Provide classes to interface with the Monocular Depth Estimation Module (MDEM)
'''
import warnings

import torch


from UTILS.io_utils import FrameIO

class MDEMInterface:
    # This class call/initialize the classes used to interface with the MDEM
    def __init__(self, model_type = "ZoeD_NK"):
        # when we initialize this class the model will be initialized
        self.zoe = self._initialize_ZOE(model_type)

    def _initialize_ZOE(self, model_type):
        # It is recommended to fetch the latest MiDaS repo via torch hub before proceeding
        torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo

        # load the model from torch hub
        if model_type == "ZoeD_N":
            zoe = torch.hub.load("isl-org/ZoeDepth", model_type, pretrained=True)
        elif model_type == "ZoeD_K":
            zoe = torch.hub.load("isl-org/ZoeDepth", model_type, pretrained=True)
        elif model_type == "ZoeD_NK":
            zoe = torch.hub.load("isl-org/ZoeDepth", model_type, pretrained=True)
        else:
            warnings.warn(f"The model type selected [{model_type}], does not exist! Using default model [ZoeD_NK]")
            zoe = torch.hub.load("isl-org/ZoeDepth", model_type, pretrained=True)

        # pass the model to the proper device
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] model loaded on {DEVICE}")
        zoe = zoe.to(DEVICE)

        return zoe


    def infer_monocular_depth_map(self, path_to_frame):
        '''
        This method infer the depth map, using ZOE, from a monocular frame
        Parameters:
        - path_to_frame: path to the frame [str]

        Returns:
        - depth_map
        '''

        # step 1: we load the frame
        image = FrameIO.load_p_img(path_to_frame, convert_to_rgb=True)

        # step 2: we infer the depth map
        depth_map = self.zoe.infer_pil(image, output_type = "pil") # as 16 bit PIL Image

        return depth_map


    def save_depth_map(self, image, saving_path, extension = None):
        '''
        This method save a depth map
        Parameters:
        - image: PIL image
        - saving_path: path to the saving folder [str]
        - extension: None or string to the desired extension [str]

        '''

        FrameIO.save_p_img(image, saving_path, extension)

    def debug(self, path_to_frame, saving_path):
        '''
        This function test all the methods of this class

        Parameters:
        - path_to_frame: path to the input image/frame
        - saving_path: path to the folder used for saving
        '''
        passed = []
        print("[DEBUG]: Testing infer method...")
        try:
            MDEMInterface.infer_monocular_depth_map(path_to_frame)
            passed.append(True)
        except ValueError as e:
            passed[-1] = False
            print(f"[DEBUG]: OPS :/ -> {e}")
        else:
            print("[DEBUG]: infer method status -> ok")

        print("[DEBUG]: Testing saving method...")
        try:
            MDEMInterface.save_depth_map(saving_path)
            passed.append(True)
        except ValueError as e:
            passed[-1] = False
            print(f"[DEBUG]: OPS :/ -> {e}")

        else:
            print("[DEBUG]: saving method status -> ok")

        










