import torch
import torch.nn.functional as F


def mae_loss(prediction, target):
    """
    Args:
        prediction: shape (B, N, Y)
        target: shape (B, N, Y)
    """
    mask = ~torch.isnan(target)
    masked_prediction = prediction[mask]
    masked_target = target[mask]
    return F.l1_loss(masked_prediction, masked_target)


def mse_loss(prediction, target):
    """
    Args:
        prediction: shape (B, N, Y)
        target: shape (B, N, Y)
    """
    mask = ~torch.isnan(target)
    masked_prediction = prediction[mask]
    masked_target = target[mask]
    return F.mse_loss(masked_prediction, masked_target)
