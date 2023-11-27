from .image_preprocessor import ImagePreprocessor
import torch
import cv2
import numpy as np


class SatellitePreprocessor(ImagePreprocessor):
    '''
            Class defaultly accepts images of 8160x8160 pixels and divides them by subimages.
        In the result preprocessed images of specified size are being stack in one
        tensor to be passed through SuperPoint model.

            For the reason of flexibility "input_size" parameter were added. What that means is
        satellite image must be divisible on "input_size". Also image must be a square-shaped. No
        tests were taken to test this additional functionality. 
    '''
    
    def __init__(self, config: dict={}):
        super().__init__()
        self.input_size = config.get('input_size', 480)


    def _slice_image(self, image, child_num: int=None):
        '''
                This function slices image into child_num parts. Also it is assumed, 
            that shape of the image is also a square. Number of childs must be a square!

            child_num: corresponds to the number of images extracted from the original image.
        '''
        if child_num is None:
            rows = (image.shape[0] // self.input_size)

        step = self.input_size

        for row in range(rows):
            for column in range(rows):
                child_image = image[row * step : (row + 1) * step, column * step : (column + 1) * step].copy()
                yield child_image


    def __call__(self, image: cv2.Mat) -> torch.Tensor:
        '''
                Function accepts, for example, image 8160x8160, slices it into 480x480 images.
            In the result we get 289 subimages in grayscale.
        '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        child_images = [
            torch.as_tensor(np.expand_dims(image, axis=0))
            for image in self._slice_image(image=image)
        ]

        return torch.stack([child_image / 255. for child_image in child_images], dim=0)