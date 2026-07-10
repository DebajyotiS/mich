"""Group B -- bandwidth ceiling. Only decisive if Group A/L show a real E_deriv, but B1
always runs and reports; B2 is a short-training capacity probe (@pytest.mark.slow)."""

from __future__ import annotations

import copy
import math

import pytest
import torch
from common import (
    _forward_and_gather,
    build_val_loader,
    detect_edges,
    dt_index,
    edge_window_mask,
    power_spectrum,
    rms,
    write_report,
)

EDGE_WINDOW = 3
EDGE_TOL = 0.05
N_OVERFIT_STEPS = 300
OVERFIT_LR = 1e-3


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, t.shape[-1])


def test_required_bandwidth(diag_val_data, diag_out_dir):
    """B1: PSD of dt_index(s_true) -- if the true derivative's required bandwidth exceeds
    max_freq, the FiLM basis cannot represent the derivative spikes regardless of Fix 1."""
    d = diag_val_data
    s_true = _flat(d["s_true"])
    max_freq, nyquist = d["max_freq"], (s_true.shape[-1] - 1) / 2.0

    D_true = dt_index(s_true, dim=-1)
    freqs, psd = power_spectrum(D_true, dim=-1)
    psd_mean = psd.mean(dim=0)

    total_power = psd_mean.sum().clamp(min=1e-12)
    cumulative = torch.cumsum(psd_mean, dim=0) / total_power
    idx_99 = int((cumulative >= 0.99).nonzero(as_tuple=True)[0][0])
    bandwidth_99 = freqs[idx_99].item()

    report = {
        "max_freq_available": max_freq,
        "nyquist": nyquist,
        "bandwidth_containing_99pct_power": bandwidth_99,
        "frac_power_above_max_freq": (psd_mean[freqs > max_freq].sum() / total_power).item(),
    }
    write_report(report, diag_out_dir, "B1_required_bandwidth")

    assert math.isfinite(bandwidth_99)


def _edge_rms(model, batch, device) -> tuple[float, float]:
    with torch.no_grad():
        gathered = _forward_and_gather(model, batch, device)
    s_hat, s_true, x_true = (
        _flat(gathered["s_hat"]),
        _flat(gathered["s_true"]),
        _flat(gathered["x_true"]),
    )
    E_deriv = dt_index(s_hat, dim=-1) - dt_index(s_true, dim=-1)
    edges = detect_edges(x_true, tol=EDGE_TOL)
    T = x_true.shape[-1]
    mask = torch.stack(
        [edge_window_mask(T, torch.cat([e["rising"], e["falling"]]), EDGE_WINDOW) for e in edges]
    )
    edge_rms = rms(E_deriv[mask]) if mask.any() else math.nan
    return edge_rms, rms(E_deriv)


def _manual_loss(model, batch, lc):
    bold = batch["bold"]
    source_position, num_sources = batch["source_position"], batch["num_sources"]
    bold_norm = (
        model.normaliser(bold, source_position, num_sources)
        if model.normaliser is not None
        else bold
    )
    time_grid = model._make_time_grid(
        B=bold.shape[0], T=bold.shape[2], device=bold.device, dtype=bold.dtype
    )
    manifest = model(bold_norm, time_grid, return_gradients=True, normalise=False)
    z_hat, dz_hat_dt = manifest.z_hat, manifest.grads

    data_loss, _, _ = model._data_loss(
        z_hat, bold_norm, source_position=source_position, num_sources=num_sources
    )
    physics_loss, _ = model._physics_loss(
        z_hat,
        dz_hat_dt,
        order=lc.order,
        lambda_smooth=0.0,
        source_position=source_position,
        num_sources=num_sources,
    )
    total = lc.lambda_data * data_loss + lc.lambda_physics * physics_loss
    if getattr(lc, "lambda_supervision", 0.0) > 0:
        sup_loss, _ = model._source_supervision_loss(z_hat, batch, source_position, num_sources)
        total = total + lc.lambda_supervision * sup_loss
    if getattr(lc, "supervise_dzdt", False):
        dzdt_loss, _ = model._derivative_supervision_loss(
            dz_hat_dt, batch, source_position, num_sources
        )
        total = total + lc.lambda_dzdt_supervision * dzdt_loss
    return total


@pytest.mark.slow
def test_capacity_probe(diag_model, diag_device, diag_out_dir):
    """B2 [short training]: overfit a deep copy of the model to a single val sample and
    measure the achievable edge-window E_deriv after overfitting. Irreducible error at
    max effort on ONE sample is a representation ceiling, not an optimization/weighting
    issue with the full training run."""
    model, full_cfg = diag_model
    lc = full_cfg.model.loss_config
    loader = build_val_loader(full_cfg)
    batch = next(iter(loader))
    single = {k: (v[:1].to(diag_device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    before_edge_rms, before_full_rms = _edge_rms(model, single, diag_device)

    model_copy = copy.deepcopy(model).to(diag_device)
    model_copy.train()
    opt = torch.optim.Adam(model_copy.parameters(), lr=OVERFIT_LR)

    losses = []
    for _ in range(N_OVERFIT_STEPS):
        total = _manual_loss(model_copy, single, lc)
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(total.item())
    model_copy.eval()

    after_edge_rms, after_full_rms = _edge_rms(model_copy, single, diag_device)

    report = {
        "n_steps": N_OVERFIT_STEPS,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "before_overfit_edge_E_deriv_rms": before_edge_rms,
        "after_overfit_edge_E_deriv_rms": after_edge_rms,
        "before_overfit_full_E_deriv_rms": before_full_rms,
        "after_overfit_full_E_deriv_rms": after_full_rms,
    }
    write_report(report, diag_out_dir, "B2_capacity_probe")

    assert math.isfinite(after_edge_rms)
    assert losses[-1] < losses[0], "overfitting a single sample should reduce its own loss"
