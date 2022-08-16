import numpy as np


def consistency_loss(mixture, music):
    music = np.clip(np.sum(music, axis=3), 0, 1)
    mixture = np.squeeze(mixture)

    xor = np.logical_xor(mixture, music)
    consistency = np.sum(xor) / np.prod(mixture.shape)

    return consistency


def diversity_loss(mixture, y_true, y_pred):
    y_pred = y_pred * mixture
    xor = np.logical_xor(y_true, y_pred)
    diversity = np.sum(xor) / np.prod(y_true.shape)

    return diversity
