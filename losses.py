import numpy as np


def mse_loss(outputs, target):
    return ((outputs - target) ** 2).sum() / len(outputs)


def mse_loss_d(outputs, target):
    return 2 * (outputs - target) / len(outputs)


def cross_entropy(model_output, target):
    if model_output.shape != target.shape:
        raise ValueError('Dimensions of model output and target do not match')
    eps = np.finfo(model_output.dtype).eps
    model_output = np.clip(model_output, eps, 1- eps)
    return -np.sum(target * np.log(model_output))/model_output.shape[0]


def cross_entropy_d(model_output, target):
    return model_output - target


mse_loss_set = (mse_loss, mse_loss_d)
crossentropy_loss_set = (cross_entropy, cross_entropy_d)
