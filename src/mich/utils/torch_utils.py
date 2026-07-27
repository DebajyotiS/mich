"""Activation/normalisation factories and elementwise activation derivatives.

The `_*_deriv` functions each compute the analytic derivative of one activation,
for use where a caller needs d/dx of an activation it applies elsewhere (e.g.
`blocks.HEINZLE_ACTIVATIONS` pairs an activation with its derivative for the
chain-rule computation in `SpatioTemporalDecoder._apply_activation_derivatives`).
Only `_softplus_deriv` is currently wired into that pairing; `_sigmoid_deriv`,
`_one_plus_softplus`, `_neg_softplus_neg`, `_neg_softplus_neg_deriv`, and
`_tanh_deriv` have no caller in `src/` and are exercised only by
`tests/test_grads.py`.
"""

import torch
from torch import nn
from torch.nn import functional as F


def get_activation(activation: str) -> nn.Module:
    """Build an `nn.Module` for a named activation function.

    Args:
        activation: Name, case-insensitive (e.g. "silu", "SiLU"). See the
            `match` cases below for the full supported set.

    Raises:
        ValueError: If `activation` does not match a supported name.
    """
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
    """Build an `nn.Module` for a named normalisation layer.

    Args:
        normalisation: Name, case-insensitive ("batchnorm", "layernorm",
            "instancenorm", "groupnorm", "none").
        input_dims: Number of channels/features the layer normalises over.
        **kwargs: `num_groups` (default 32), used only when
            normalisation="groupnorm".

    Raises:
        ValueError: If `normalisation` does not match a supported name.
    """
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
    """d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))"""
    s = torch.sigmoid(x)
    return s * (1.0 - s)


def _softplus_deriv(x: torch.Tensor) -> torch.Tensor:
    """d/dx softplus(x) = sigmoid(x)"""
    return torch.sigmoid(x)


def _one_plus_softplus(x: torch.Tensor) -> torch.Tensor:
    """1 + softplus(x) -- smooth, bounded below by 1."""
    return 1.0 + F.softplus(x)


def _neg_softplus_neg(x: torch.Tensor) -> torch.Tensor:
    """-softplus(-x)  -- non-positive, smooth"""
    return -F.softplus(-x)


def _neg_softplus_neg_deriv(x: torch.Tensor) -> torch.Tensor:
    """d/dx [-softplus(-x)] = sigmoid(x) - 1"""
    return torch.sigmoid(x) - 1.0


def _tanh_deriv(x: torch.Tensor) -> torch.Tensor:
    """d/dx tanh(x) = 1 - tanh^2(x)"""
    t = torch.tanh(x)
    return 1.0 - t * t
