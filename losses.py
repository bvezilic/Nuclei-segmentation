import torch
import torch.nn.functional as F


def dice_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Soft dice loss = 2*|A∩B| / |A|+|B|
    """
    numerator = 2 * (input * target).sum((1, 2, 3))
    denominator = (input + target).sum((1, 2, 3))

    return (1 - numerator / denominator).mean()


def bce_and_dice(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combines both BCE and Dice loss
    """
    bce_loss = F.binary_cross_entropy(input=input, target=target)
    d_loss = dice_loss(input=input, target=target)

    return bce_loss + d_loss
