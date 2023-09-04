'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 30/08/23


Description:
Provide the function used to load and save images, files, etc...
'''
import warnings
from PIL import Image

class FrameIO:
    def load_p_img(self, path_to_img, convert_to_rgb = False):
        '''
        This function load an img using the Image from PIL

        Parameters:
        - path_to_img: path of the img [str]

        Return:
        - PIL image
        '''
        if convert_to_rgb:
            return Image.open(path_to_img).convert("RGB")
        else:
            return Image.open(path_to_img)

    def save_p_img(self, img, saving_path, extension = None):
        '''
        This function save a PIL img

        Parameters:
        - img: image to be saved [PIL image]
        - saving_path: path for saving the img [str]
        - extension: the extension of the img

        Return:
        - True (if success)
        - False (if failed)
        '''
        try:
            if extension is None:
                # no extension has been provided, this assumes that it is in the saving path
                warnings.warn("No extension has been provided")
                img.save(saving_path)
                return True
            else:
                saving_path_with_extension = saving_path + extension
                img.save(saving_path_with_extension)
                return True
        except ValueError as e:
            print(f"Error while saving: {e}")



