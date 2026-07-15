"""Group O -- origin of the delay. Only decisive if Group L shows a real operator gap,
but always runs and reports."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from common import _forward_and_gather, build_val_loader, dt_index, rms, write_report, xcorr_lag

from mich.data.balloon import (
    CortexLayer,
    HaemodynamicConstants,
    HaemodynamicState,
    simulate_cortex,
)


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, t.shape[-1])


@pytest.fixture(scope="session")
def diag_val_data_ablated(diag_model, diag_device) -> dict[str, torch.Tensor]:
    """Same as diag_val_data but with the decoder-input features (`u`, the output of
    temporal_mixing) replaced by their temporal mean before the decoder -- isolates
    whether the time-varying TCN-over-BOLD path is what carries the operator delay."""
    from collections import defaultdict

    model, full_cfg = diag_model
    loader = build_val_loader(full_cfg)

    def _ablate(u: torch.Tensor) -> torch.Tensor:
        return u.mean(dim=1, keepdim=True).expand_as(u)

    collected: dict[str, list[torch.Tensor]] = defaultdict(list)
    for batch in loader:
        gathered = _forward_and_gather(model, batch, diag_device, xmix_transform=_ablate)
        for k, v in gathered.items():
            collected[k].append(v.cpu())
    return {k: torch.cat(v, dim=0) for k, v in collected.items()}


def test_u_ablation(diag_val_data, diag_val_data_ablated, diag_out_dir):
    """O1: if ablating the time-varying decoder input collapses lag(D_partial, D_emit)
    toward 0, the operator delay is carried by that path (TCN-over-BOLD); s_hat's own
    value fit should also degrade under the ablation, confirming the path carries real
    (not incidental) content."""
    d, d_abl = diag_val_data, diag_val_data_ablated

    def _lag_gap(data):
        s_hat, Dp_s = _flat(data["s_hat"]), _flat(data["Dp_s"])
        D_emit = dt_index(s_hat, dim=-1)
        return xcorr_lag(Dp_s, D_emit).mean().item()

    lag_normal = _lag_gap(d)
    lag_ablated = _lag_gap(d_abl)

    s_hat_true_rmse_normal = rms(_flat(d["s_hat"]) - _flat(d["s_true"]))
    s_hat_true_rmse_ablated = rms(_flat(d_abl["s_hat"]) - _flat(d_abl["s_true"]))

    report = {
        "lag_Dp_s_vs_D_emit_normal": lag_normal,
        "lag_Dp_s_vs_D_emit_ablated": lag_ablated,
        "s_value_rmse_normal": s_hat_true_rmse_normal,
        "s_value_rmse_ablated": s_hat_true_rmse_ablated,
        "s_value_rmse_degradation_ratio": s_hat_true_rmse_ablated / (s_hat_true_rmse_normal + 1e-8),
    }
    write_report(report, diag_out_dir, "O1_u_ablation")

    assert math.isfinite(lag_normal) and math.isfinite(lag_ablated)
    # Sanity: ablating a real input path must not IMPROVE the value fit.
    assert report["s_value_rmse_degradation_ratio"] > 0.9


def test_delay_vs_hemodynamic_lag(diag_val_data, diag_out_dir):
    """O2: simulate a single neural step through the run's own balloon constants and
    compare its rise-lag in s to L1's measured operator lag lag(D_partial, D_emit).
    dt=1.0 in this simulation matches the model's sample spacing, so both are already in
    the same units (samples) -- no conversion needed. Comparable magnitude corroborates a
    BOLD-path origin for the operator delay; not comparable -> the delay is elsewhere."""
    d = diag_val_data
    s_hat, Dp_s = _flat(d["s_hat"]), _flat(d["Dp_s"])
    D_emit = dt_index(s_hat, dim=-1)
    operator_lag_samples = xcorr_lag(Dp_s, D_emit).mean().item()

    haemo = d["haemo"]
    constants = HaemodynamicConstants(
        kappa=haemo["kappa"], gamma=haemo["gamma"], alpha=haemo["alpha"], E0=d["E0"], V0=d["V0"]
    )
    T = s_hat.shape[-1]
    onset = 10
    x_step = np.concatenate([np.zeros(onset), np.ones(T - onset)])
    layer = CortexLayer(
        depth=0,
        tau=haemo["tau"],
        state=HaemodynamicState(x=0.0, s=0.0, f=1.0, v=1.0, q=1.0),
        lambda_d=0.0,
        drain_from=None,
    )
    out = simulate_cortex(
        [layer], constants, [x_step], dt=1.0, tau_d=haemo["tau_d"], order=d["order"]
    )
    s_response = out[0]["s"]

    post_onset = s_response[onset:]
    peak_idx = int(np.argmax(post_onset))
    half_max = post_onset.max() / 2.0
    half_idx_candidates = np.nonzero(post_onset >= half_max)[0]
    half_idx = int(half_idx_candidates[0]) if half_idx_candidates.size else peak_idx

    report = {
        "operator_lag_samples": operator_lag_samples,
        "hemodynamic_time_to_peak_s_samples": peak_idx,
        "hemodynamic_time_to_half_max_s_samples": half_idx,
        "ratio_operator_lag_to_half_max": abs(operator_lag_samples) / max(half_idx, 1),
    }
    write_report(report, diag_out_dir, "O2_delay_vs_hemodynamic_lag")

    assert math.isfinite(operator_lag_samples)
    assert peak_idx >= 0
