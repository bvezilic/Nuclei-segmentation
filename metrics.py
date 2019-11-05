import torch


def iou_score(inputs: torch.Tensor,
              masks: torch.Tensor,
              epsilon: float = 1e-6) -> torch.Tensor:
    """
    Computes Intersection-over-Union for input images and maks.

    Credit to: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy

    Args:
        inputs: 3D Tensor (batch_size, channels=1, height, width)
        masks: 3D Tensor (batch_size, channels=1, height, width)
        epsilon: Defaults to 1e-6

    Returns:
        IoU score
    """
    intersection = (inputs * masks).sum((1, 2, 3))
    union = (inputs + masks).sum((1, 2, 3))

    iou = ((intersection + epsilon) / (union + epsilon)).mean()

    return iou
