import cv2
import numpy as np


class Rescale:
    def __init__(self, output_size: int):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.output_size, self.output_size), cv2.INTER_AREA)
