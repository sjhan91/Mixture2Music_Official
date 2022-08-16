import torch.nn.functional as F
from torch import nn


def bce_loss(y_true, y_pred):
    return nn.BCEWithLogitsLoss()(y_pred, y_true)


def p_losses(y_true, y_pred, loss_type="l1"):
    if loss_type == "l1":
        loss = F.l1_loss(y_true, y_pred)
    elif loss_type == "l2":
        loss = F.mse_loss(y_true, y_pred)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(y_true, y_pred)
    else:
        raise NotImplementedError()

    return loss
