"""Tests for get_activation and get_normalisation factory functions in torch_utils."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mich.utils.torch_utils import get_activation, get_normalisation

# -------------------------
# get_activation
# -------------------------


@pytest.mark.parametrize(
    "name, expected_type",
    [
        ("relu", nn.ReLU),
        ("ReLU", nn.ReLU),  # case-insensitive
        ("sigmoid", nn.Sigmoid),
        ("silu", nn.SiLU),
        ("SiLU", nn.SiLU),
        ("tanh", nn.Tanh),
        ("gelu", nn.GELU),
        ("lrlu", nn.LeakyReLU),
        ("prelu", nn.PReLU),
        ("elu", nn.ELU),
        ("selu", nn.SELU),
        ("hardsigmoid", nn.Hardsigmoid),
        ("hardtanh", nn.Hardtanh),
        ("hardswish", nn.Hardswish),
        ("logsigmoid", nn.LogSigmoid),
        ("softplus", nn.Softplus),
        ("softsign", nn.Softsign),
        ("tanhshrink", nn.Tanhshrink),
        ("none", nn.Identity),
        ("NONE", nn.Identity),
    ],
)
def test_get_activation_returns_correct_module(name, expected_type):
    act = get_activation(name)
    assert isinstance(act, expected_type)


def test_get_activation_returns_new_instance_each_call():
    a1 = get_activation("relu")
    a2 = get_activation("relu")
    assert a1 is not a2


def test_get_activation_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported activation"):
        get_activation("swish_custom")


def test_get_activation_forward_pass_runs():
    """Each registered activation should be callable on a real tensor."""
    names = ["relu", "sigmoid", "silu", "tanh", "gelu", "elu", "selu", "none", "softplus"]
    x = torch.randn(4, 8)
    for name in names:
        act = get_activation(name)
        y = act(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all(), f"{name} produced non-finite output"


# -------------------------
# get_normalisation
# -------------------------


@pytest.mark.parametrize(
    "name, expected_type",
    [
        ("batchnorm", nn.BatchNorm1d),
        ("BatchNorm", nn.BatchNorm1d),  # case-insensitive
        ("layernorm", nn.LayerNorm),
        ("LayerNorm", nn.LayerNorm),
        ("instancenorm", nn.InstanceNorm1d),
        ("none", nn.Identity),
    ],
)
def test_get_normalisation_returns_correct_module(name, expected_type):
    norm = get_normalisation(name, input_dims=16)
    assert isinstance(norm, expected_type)


def test_get_normalisation_groupnorm_returns_correct_module():
    # input_dims must be divisible by num_groups (default 32)
    norm = get_normalisation("groupnorm", input_dims=32)
    assert isinstance(norm, nn.GroupNorm)


def test_get_normalisation_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported normalisation"):
        get_normalisation("spectral", input_dims=16)


def test_get_normalisation_groupnorm_uses_num_groups_kwarg():
    norm = get_normalisation("groupnorm", input_dims=16, num_groups=4)
    assert isinstance(norm, nn.GroupNorm)
    assert norm.num_groups == 4


def test_get_normalisation_groupnorm_default_num_groups():
    # Default is 32, but input_dims=32 to satisfy GroupNorm divisibility
    norm = get_normalisation("groupnorm", input_dims=32)
    assert isinstance(norm, nn.GroupNorm)
    assert norm.num_groups == 32


def test_get_normalisation_forward_pass_runs():
    """Each normalisation type should forward without error."""
    x = torch.randn(8, 16)  # [batch, features]
    for name in ("batchnorm", "layernorm", "none"):
        norm = get_normalisation(name, input_dims=16)
        norm.eval()
        y = norm(x)
        assert y.shape == x.shape
