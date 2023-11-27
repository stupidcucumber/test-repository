from .image_preprocessor import ImagePreprocessor
import numpy as np
import cv2
import torch


class BasePreprocessor(ImagePreprocessor):
    def __init__(self, config: dict={}):
        super().__init__()
        self.input_size = config.get('input_size', 480)


    def __call__(self, image: cv2.Mat) -> torch.Tensor:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, dsize=(self.input_size, self.input_size))
        image = np.asarray([image], dtype=np.float32) / 255.
        
        return torch.as_tensor(np.expand_dims(image, 0))