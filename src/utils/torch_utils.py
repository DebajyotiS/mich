import torch
from torch.nn import functional as F


def get_activation(activation: str):
    match activation.casefold():
        case "relu":
            return F.relu
        case "sigmoid":
            return torch.sigmoid
        case "silu":
            return F.silu
        case "tanh":
            return torch.tanh
        case "gelu":
            return F.gelu
