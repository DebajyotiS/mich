"""Group S -- structural guardrails on a tiny fixture. No checkpoint needed.

S1/S2 establish that the operator gap (E_op in the spec) is possible at all: the
decoder's analytic `dz_hat_dt` is a PARTIAL derivative (holds the decoder-input features
constant in t), so it only equals the TOTAL derivative of the emitted signal when that
input genuinely is constant in t. S3 rules out an autograd bug in the FiLM-time-gradient
machinery (`_gamma_beta_time_grads`) that `dz_hat_dt` is built from, independent of the
rest of the pipeline.
"""

from __future__ import annotations

import pytest
import torch
from common import dt_index
from mich.models.blocks import SpatioTemporalDecoder


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _tiny_decoder(*, num_freqs: int = 4, max_freq: float = 3.0) -> SpatioTemporalDecoder:
    """Single-signal ("s"), single-layer decoder -- fast, but exercises the same
    FiLM/decode-dt machinery as the full 7-signal, multi-layer model.

    The output heads are zero-init by construction (see blocks.py's
    `_init_heinzle_output_bias`-adjacent init in `SpatioTemporalDecoder.__init__`, meant
    to bootstrap real training) -- with zero head weight, z_pre is a constant (just the
    bias) and every derivative is trivially zero regardless of the input. Re-randomizing
    every parameter after construction is required for these gap tests to exercise
    anything.
    """
    dec = SpatioTemporalDecoder(
        cin=3,
        c_dec=5,
        c_film=3,
        out_channels=1,
        activation="silu",
        L=1,
        temporal_embedding_config=dict(num_freqs=num_freqs, max_freq=max_freq),
        temporal_film_config=dict(
            embed_dim=2 * num_freqs, hidden_dim=8, activation="silu", c_dec=3
        ),
        signals=["s"],
    )
    with torch.no_grad():
        for p in dec.parameters():
            torch.nn.init.normal_(p, std=0.3)
    return dec


def _rel_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    num = (a - b).float().pow(2).mean().sqrt()
    den = b.float().pow(2).mean().sqrt().clamp(min=1e-8)
    return (num / den).item()


def test_partial_neq_total_on_varying_input():
    dec = _tiny_decoder()
    B, T, L, cin, H, W = 2, 20, 1, 3, 4, 4
    x = torch.randn(B, T, L, cin, H, W)  # decoder input varies freely with t
    t = torch.linspace(0.0, 1.0, T).unsqueeze(0).expand(B, -1)

    m = dec(x, t, return_gradients=True)
    D_partial = m.grads[:, 0]  # [B, L, T, H, W]
    D_total = dt_index(m.z_hat[:, 0], dim=2)  # T is dim=2 in [B, L, T, H, W]

    gap = _rel_gap(D_partial, D_total)
    assert gap > 0.1, f"expected a clear operator gap on time-varying input, got rel_gap={gap}"


def test_constant_input_removes_gap():
    dec = _tiny_decoder()
    B, T, L, cin, H, W = 2, 200, 1, 3, 4, 4
    x_const = torch.randn(B, 1, L, cin, H, W).expand(B, T, L, cin, H, W)  # constant in T
    t = torch.linspace(0.0, 1.0, T).unsqueeze(0).expand(B, -1)

    m = dec(x_const, t, return_gradients=True)
    D_partial = m.grads[:, 0]
    D_total = dt_index(m.z_hat[:, 0], dim=2)

    gap = _rel_gap(D_partial, D_total)
    assert gap < 0.05, (
        f"expected the operator gap to vanish when the decoder input is constant in t "
        f"(z then varies with t only through FiLM, which dz_hat_dt is exact for), got "
        f"rel_gap={gap}"
    )


def test_film_time_grads_match_fd():
    dec = _tiny_decoder(num_freqs=4, max_freq=3.0)
    t = torch.linspace(0.1, 0.9, 5).unsqueeze(0)  # [1, 5], arbitrary (not required to be [0,1])

    dgamma_dt, dbeta_dt = dec._gamma_beta_time_grads(t)

    eps = 1e-4
    gamma_p, beta_p = dec._gamma_beta(t + eps)
    gamma_m, beta_m = dec._gamma_beta(t - eps)
    fd_dgamma = (gamma_p - gamma_m) / (2 * eps)
    fd_dbeta = (beta_p - beta_m) / (2 * eps)

    assert torch.isfinite(dgamma_dt).all() and torch.isfinite(dbeta_dt).all()
    assert torch.allclose(dgamma_dt, fd_dgamma, atol=1e-3, rtol=5e-2)
    assert torch.allclose(dbeta_dt, fd_dbeta, atol=1e-3, rtol=5e-2)
