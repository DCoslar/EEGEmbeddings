import torch


def fro_loss(output, target):
    return torch.sum((output - target).flatten()**2)
