"""Is the ~1-sample x_hat-vs-x_true lag real, or is it integer-argmax xcorr snapping to
the nearest sample on a signal that's actually at/near the resolution floor?

The `-1.0` with zero variance from the original A2/L1 report is exactly what a
quantized (integer-only) estimator looks like whether or not there's a real delay
underneath, so it settles nothing on its own. This file re-measures the same offset with
three independent sub-sample estimators (parabolic interpolation, cross-spectrum phase
slope, band-limited upsampling) and checks they agree, checks per-edge spread (a real
delay is consistent edge to edge; a floor artifact scatters), and checks a noise-free
calibration pair (does the offset survive when observation noise is removed at the source
voxel?). This is the gate: origin-hunting (TCN impulse probe, redone O2, Fix-1 retrain)
is only worth doing if this comes back as a genuine, non-floor delay.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from common import (
    build_val_loader,
    detect_edges,
    parabolic_lag_at_edge,
    parabolic_subsample_lag,
    phase_slope_lag,
    upsampled_xcorr_lag,
    write_report,
    xcorr_lag,
)
from mich.data.balloon import CortexLayer, HaemodynamicConstants, HaemodynamicState, simulate_cortex

EDGE_TOL = 0.05
EDGE_WINDOW = 8


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, t.shape[-1])


def _gaussian_bump(t: np.ndarray, center: float, width: float = 3.0) -> np.ndarray:
    return np.exp(-0.5 * ((t - center) / width) ** 2)


def test_sign_convention_and_subsample_accuracy_self_check():
    """No checkpoint needed: verify all three estimators (a) agree with xcorr_lag's stated
    sign convention and (b) actually recover known FRACTIONAL delays on a smooth synthetic
    signal, not just integer ones -- this is what justifies trusting them on real data
    below, rather than assuming the derivation (esp. phase_slope_lag's sign) is right.
    """
    T = 200
    t = np.arange(T, dtype=np.float64)
    true = torch.from_numpy(_gaussian_bump(t, center=100.0)).float()

    for true_delay in (0.0, 0.5, 1.0, 1.37, 2.0, -1.5):
        pred = torch.from_numpy(_gaussian_bump(t, center=100.0 + true_delay)).float()

        integer_lag = xcorr_lag(pred, true).item()
        parabolic = parabolic_subsample_lag(pred, true).item()
        upsampled = upsampled_xcorr_lag(pred, true, factor=20).item()
        phase = phase_slope_lag(pred, true, freq_cutoff=0.45)

        assert abs(integer_lag - round(true_delay)) <= 1, (true_delay, integer_lag)
        assert abs(parabolic - true_delay) < 0.15, (true_delay, parabolic)
        assert abs(upsampled - true_delay) < 0.15, (true_delay, upsampled)
        assert abs(phase["delay"] - true_delay) < 0.15, (true_delay, phase)


def test_subsample_lag_agreement(diag_val_data, diag_out_dir):
    """The actual gate. Re-measure x_hat vs x_true with three independent methods."""
    d = diag_val_data
    x_hat, x_true = _flat(d["x_hat"]), _flat(d["x_true"])

    integer_lags = xcorr_lag(x_hat, x_true)
    parabolic_lags = parabolic_subsample_lag(x_hat, x_true)
    upsampled_lags = upsampled_xcorr_lag(x_hat, x_true, factor=10)
    phase = phase_slope_lag(x_hat, x_true, freq_cutoff=0.4)

    report = {
        "integer_xcorr": {"mean": integer_lags.mean().item(), "std": integer_lags.std().item()},
        "parabolic": {"mean": parabolic_lags.mean().item(), "std": parabolic_lags.std().item()},
        "upsampled_10x": {"mean": upsampled_lags.mean().item(), "std": upsampled_lags.std().item()},
        "phase_slope": phase,
    }
    write_report(report, diag_out_dir, "LS1_subsample_agreement")

    for key in ("integer_xcorr", "parabolic", "upsampled_10x"):
        assert math.isfinite(report[key]["mean"])
    assert math.isfinite(phase["delay"])


def test_subsample_lag_per_edge_spread(diag_val_data, diag_out_dir):
    """Per-edge parabolic lag distribution: tight clustering = real systematic delay;
    wide scatter (e.g. roughly uniform over [-0.5, 0.5]) = floor artifact."""
    d = diag_val_data
    x_hat, x_true = _flat(d["x_hat"]), _flat(d["x_true"])
    edges = detect_edges(x_true, tol=EDGE_TOL)

    lags = []
    for i, e in enumerate(edges):
        for edge in torch.cat([e["rising"], e["falling"]]).tolist():
            lag = parabolic_lag_at_edge(x_hat[i], x_true[i], edge, EDGE_WINDOW)
            if lag is not None:
                lags.append(lag)

    lags_t = torch.tensor(lags)
    report = {
        "n_edges": len(lags),
        "mean": lags_t.mean().item(),
        "std": lags_t.std().item(),
        "median": lags_t.median().item(),
        "p10": lags_t.quantile(0.1).item(),
        "p90": lags_t.quantile(0.9).item(),
        "frac_within_0.3_of_mean": ((lags_t - lags_t.mean()).abs() < 0.3).float().mean().item(),
    }
    write_report(report, diag_out_dir, "LS2_per_edge_spread")

    assert len(lags) > 100


def test_noise_free_calibration(diag_model, diag_device, diag_out_dir):
    """Does the lag survive with observation noise removed at the source voxel? Rather
    than trusting a synthetic scratch simulation, replicate the actual generator's own
    numerics (scripts/run_sim.py::_run_haemo_and_bold): upsample x_true by haemo_ratio via
    repeat, integrate the balloon ODE at haemo_dt, downsample back by haemo_ratio -- same
    method the real ground truth s/f/v/q were produced with, just without adding BOLD
    noise, and only at the source voxel (the rest of the spatial field keeps its real,
    noisy context so the CNN encoder still sees a realistic surround; PSF blur is a purely
    spatial operation and cannot introduce a temporal shift, so skipping it here does not
    bias the lag measurement).
    """
    import sys as _sys
    from pathlib import Path as _Path

    scripts_dir = _Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(scripts_dir))

    model, full_cfg = diag_model
    device = diag_device
    loader = build_val_loader(full_cfg)
    batch = next(iter(loader))

    # Pull sim config the same way SyntheticDataModule.sim_config does, from the same file.
    import json

    import h5py

    h5_path = full_cfg.datamodule.data.path
    with h5py.File(str(h5_path), "r") as f:
        sim_cfg = json.loads(f["meta"].attrs["config"])["simulation"]
    dt_sim, haemo_dt = float(sim_cfg["dt"]), float(sim_cfg["haemo_dt"])
    haemo_ratio = round(dt_sim / haemo_dt)

    haemo = full_cfg.model.haemo
    constants = HaemodynamicConstants(
        kappa=haemo.kappa,
        gamma=haemo.gamma,
        alpha=haemo.alpha,
        E0=full_cfg.model.acquisition.E0,
        V0=full_cfg.model.V0,
    )
    acquisition = model._current_acquisition()
    V0 = model._physio("V0")

    B = batch["bold"].shape[0]
    src_h = batch["source_position"][..., 0, 0].long()
    src_w = batch["source_position"][..., 0, 1].long()
    x_true_full = batch["neural"][:, 0]  # [B, T, H, W], single layer

    bold_clean_batch = batch["bold"].clone().float()
    for i in range(B):
        h, w = int(src_h[i]), int(src_w[i])
        x_row = x_true_full[i, :, h, w].numpy().astype(np.float64)
        x_up = np.repeat(x_row, haemo_ratio)

        layer = CortexLayer(
            depth=0,
            tau=float(haemo.tau),
            state=HaemodynamicState(x=0.0, s=0.0, f=1.0, v=1.0, q=1.0),
            lambda_d=0.0,
            drain_from=None,
        )
        out = simulate_cortex(
            [layer],
            constants,
            [x_up],
            dt=haemo_dt,
            tau_d=float(haemo.tau_d),
            order=str(full_cfg.model.loss_config.order),
        )
        v_clean = torch.from_numpy(out[0]["v"][::haemo_ratio]).float()
        q_clean = torch.from_numpy(out[0]["q"][::haemo_ratio]).float()
        bold_clean = model._compute_bold(v_clean, q_clean, acquisition, V0)
        bold_clean_batch[i, 0, :, h, w] = bold_clean

    batch_clean = dict(batch)
    batch_clean["bold"] = bold_clean_batch

    from common import _forward_and_gather

    gathered_noisy = _forward_and_gather(model, batch, device)
    gathered_clean = _forward_and_gather(model, batch_clean, device)

    x_hat_noisy = _flat(gathered_noisy["x_hat"])
    x_hat_clean = _flat(gathered_clean["x_hat"])
    x_true = _flat(gathered_noisy["x_true"])

    report = {
        "n_samples": B,
        "noisy_source": {
            "integer": xcorr_lag(x_hat_noisy, x_true).mean().item(),
            "parabolic": parabolic_subsample_lag(x_hat_noisy, x_true).mean().item(),
        },
        "noise_free_source": {
            "integer": xcorr_lag(x_hat_clean, x_true).mean().item(),
            "parabolic": parabolic_subsample_lag(x_hat_clean, x_true).mean().item(),
        },
    }
    write_report(report, diag_out_dir, "LS3_noise_free_calibration")

    assert math.isfinite(report["noise_free_source"]["parabolic"])
