'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 19/09/23

Description:
Provide function used for image processing
'''

# Computational Module
import numpy as np

class ImageProc:
    def min_max_normalization(self, image):
        '''
        This function normalize an image using min-max paradigm
        :param image:
        :return: normalize_image
        '''
        min_img = np.amin(image)
        max_img = np.amax(image)
        normalized_image = (image - min_img) / (max_img - min_img)
        return normalized_image
