import torch
from torch import nn
from torch.nn import functional as F


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


def _sigmoid_deriv(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1.0 - s)


def _softplus_deriv(x: torch.Tensor) -> torch.Tensor:
    """d/dx softplus(x) = sigmoid(x)"""
    return torch.sigmoid(x)


def _one_plus_softplus(x: torch.Tensor) -> torch.Tensor:
    return 1.0 + F.softplus(x)


def _neg_softplus_neg(x: torch.Tensor) -> torch.Tensor:
    """-softplus(-x)  — non-positive, smooth"""
    return -F.softplus(-x)


def _neg_softplus_neg_deriv(x: torch.Tensor) -> torch.Tensor:
    """d/dx [-softplus(-x)] = sigmoid(x) - 1"""
    return torch.sigmoid(x) - 1.0


def _tanh_deriv(x: torch.Tensor) -> torch.Tensor:
    """d/dx tanh(x) = 1 - tanh^2(x)"""
    t = torch.tanh(x)
    return 1.0 - t * t
