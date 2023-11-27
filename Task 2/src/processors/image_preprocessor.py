import cv2
import torch

class ImagePreprocessor():
    '''
            Abstract class that defines what class of the preprocessing image should look like.
        The only method that needs to be implemented is __call__() that accepts image (preferrably of
        type cv2.Mat).
    '''

    def __call__(self, image: cv2.Mat) -> torch.Tensor:
        raise NotImplementedError('This function must be defined in the child class!')