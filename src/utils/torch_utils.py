import torch
from torch import nn


def get_activation(activation: str) -> nn.Module:
    match activation.casefold():
        case "relu":
            return nn.ReLU()
        case "sigmoid":
            return nn.Sigmoid()
        case "silu":
            return nn.SiLU()
        case "tanh":
            return nn.Tanh()
        case "gelu":
            return nn.GELU()
        case "lrlu":
            return nn.LeakyReLU()
        case "prelu":
            return nn.PReLU()
        case "elu":
            return nn.ELU()
        case "selu":
            return nn.SELU()
        case "glu":
            return nn.GLU()
        case "hardsigmoid":
            return nn.Hardsigmoid()
        case "hardtanh":
            return nn.Hardtanh()
        case "hardswish":
            return nn.Hardswish()
        case "logsigmoid":
            return nn.LogSigmoid()
        case "softplus":
            return nn.Softplus()
        case "softsign":
            return nn.Softsign()
        case "tanhshrink":
            return nn.Tanhshrink()
        case "none":
            return nn.Identity()
        case _:
            raise ValueError(f"Unsupported activation: {activation}")


def get_normalisation(normalisation: str, input_dims: int, **kwargs):
    match normalisation.casefold():
        case "batchnorm":
            return torch.nn.BatchNorm1d(input_dims)
        case "layernorm":
            return torch.nn.LayerNorm(input_dims)
        case "instancenorm":
            return torch.nn.InstanceNorm1d(input_dims)
        case "groupnorm":
            num_groups = kwargs.get("num_groups", 32)
            return torch.nn.GroupNorm(num_groups, input_dims)
        case "none":
            return torch.nn.Identity()
        case _:
            raise ValueError(f"Unsupported normalisation: {normalisation}")
