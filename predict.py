import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms, ToTensor, ToPILImage

from model import UNet
from transform import Rescale


class Predictor:
    """
    Wrapper class for UNet model to run prediction. Pre-processing and post-processing steps are included.
    """
    def __init__(self, model: UNet):
        self.model = model

        # Pre-processing for input images
        self.transforms = transforms.Compose([
            Rescale(256),
            ToTensor()])
        # Post-processing for output of the model
        self.output_transforms = transforms.Compose([
            ToPILImage()])

    def __call__(self, image: np.ndarray) -> Image:
        """
        Performs:
            1. pre-processing: resize, scale [0, 1], transformation to channel-first Tensor
            2. forward pass
            3. post-processing: conversion to PIL Image

        Args:
            image: (np.ndarray) 2D or 3D image

        Returns:
            mask: (PIL.Image) Segmented image
        """
        with torch.no_grad():
            input = self.transforms(image)
            output = self.model(input.unsqueeze(0))  # Add batch dim
            mask = self.output_transforms(output.squeeze(0))  # Remove batch dim

            return mask
