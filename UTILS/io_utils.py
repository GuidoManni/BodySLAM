'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 04/09/23


Description:
Provide the function used to load and save images, files, etc...
'''
import warnings


from PIL import Image
import cv2

class FrameIO:
    def load_p_img(self, path_to_img, convert_to_rgb = False):
        '''
        This function load an img using the Image from PIL

        Parameters:
        - path_to_img: path of the img [str]
        - convert_to_rgb: Flag value [bool]

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


    def load_cv2_img(self, path_to_img, convert_to_rgb = False):
        '''
        This function load an img using the opencv-python lib

        Parameters:
        - path_to_img: path of the img [str]
        - convert_to_rgb: Flag value [bool]

        Return:
        - cv2 img
        '''
        image = cv2.imread(path_to_img)

        if convert_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def save_cv2_img(self, img, saving_path, extension = None):
        '''
        This function save a cv2 img

        Parameters:
        - img: image to be saved [cv2 image]
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
                cv2.imwrite(saving_path, img)
                return True
            else:
                saving_path_with_extension = saving_path + extension
                cv2.imwrite(saving_path_with_extension, img)
                return True
        except ValueError as e:
            print(f"Error while saving: {e}")




