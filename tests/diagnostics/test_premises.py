"""Group P -- premises. Run first: if these fail, the rest of the attribution doesn't
mean what it claims to (see spec Section 8)."""

from __future__ import annotations

import math

import torch
from common import detect_edges, dt_physical, edge_window_mask, rms, write_report, xcorr_lag

from mich.data.balloon import (
    CortexLayer,
    HaemodynamicConstants,
    HaemodynamicState,
    balloon_derivatives,
)

EDGE_WINDOW = 3
EDGE_TOL = 0.05


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, t.shape[-1])


def _edge_mask_batch(x_true: torch.Tensor, window: int) -> torch.Tensor:
    T = x_true.shape[-1]
    edges = detect_edges(x_true, tol=EDGE_TOL)
    return torch.stack(
        [edge_window_mask(T, torch.cat([e["rising"], e["falling"]]), window) for e in edges]
    )


def test_x_slaving(diag_val_data, diag_out_dir):
    """P1: does x_hat match the RHS the physics residual actually optimizes it against,
    overall and specifically inside edge windows? Where it breaks near edges, part of the
    overshoot is x's own output representation, not the s-derivative's."""
    d = diag_val_data
    x_hat, Dp_s, s_hat, f_hat, x_true = (
        _flat(d["x_hat"]),
        _flat(d["Dp_s"]),
        _flat(d["s_hat"]),
        _flat(d["f_hat"]),
        _flat(d["x_true"]),
    )
    kappa, gamma = d["kappa"], d["gamma"]

    x_rhs = Dp_s + kappa * s_hat + gamma * (f_hat - 1.0)
    resid = x_hat - x_rhs

    edge_mask = _edge_mask_batch(x_true, EDGE_WINDOW)
    report = {
        "overall_rms": rms(resid),
        "edge_window_rms": rms(resid[edge_mask]),
        "plateau_rms": rms(resid[~edge_mask]),
        "edge_window_samples": int(edge_mask.sum()),
        "plateau_samples": int((~edge_mask).sum()),
    }
    write_report(report, diag_out_dir, "P1_x_slaving")

    assert math.isfinite(report["overall_rms"])
    assert math.isfinite(report["edge_window_rms"])


def test_s_in_phase(diag_val_data, diag_out_dir):
    """P2: s_hat should be in phase with s_true (near-zero lag) and close in value."""
    d = diag_val_data
    s_hat, s_true = _flat(d["s_hat"]), _flat(d["s_true"])

    lags = xcorr_lag(s_hat, s_true)
    report = {
        "lag_mean_samples": lags.mean().item(),
        "lag_std_samples": lags.std().item(),
        "value_rmse": rms(s_hat - s_true),
    }
    write_report(report, diag_out_dir, "P2_s_in_phase")

    assert math.isfinite(report["lag_mean_samples"])
    assert math.isfinite(report["value_rmse"])


def test_latent_value_error(diag_val_data, diag_out_dir):
    """P3: RMS of E_value -- expected tiny; if not, the "latents are perfect" premise
    behind the whole decomposition is wrong (see spec Section 8)."""
    d = diag_val_data
    s_hat, s_true, f_hat, f_true = (
        _flat(d["s_hat"]),
        _flat(d["s_true"]),
        _flat(d["f_hat"]),
        _flat(d["f_true"]),
    )
    kappa, gamma = d["kappa"], d["gamma"]

    e_value = kappa * (s_hat - s_true) + gamma * (f_hat - f_true)
    report = {
        "e_value_rms": rms(e_value),
        "s_value_rmse": rms(s_hat - s_true),
        "f_value_rmse": rms(f_hat - f_true),
    }
    write_report(report, diag_out_dir, "P3_latent_value_error")

    assert math.isfinite(report["e_value_rms"])


def test_fd_operator_matches_analytic_derivative_on_synthetic_signal():
    """P4a [assert]: `dt_physical` implementation self-test on a signal with a known
    closed-form derivative, independent of any real data's timescales -- this is the part
    of "operator sanity" that should be a tight, unconditional pass/fail. Kept separate
    from P4b (real ground truth) because the real haemodynamic signal's own timescale can
    make naive finite-differencing intrinsically lossy (see P4b) for reasons that have
    nothing to do with whether the operator is implemented correctly.
    """
    T = 500
    t = torch.linspace(0.0, 1.0, T)
    freq = 3.0
    z = torch.sin(2 * torch.pi * freq * t).unsqueeze(0)  # [1, T], dt_physical is per-index
    analytic = (2 * torch.pi * freq * torch.cos(2 * torch.pi * freq * t) / (T - 1)).unsqueeze(0)

    fd = dt_physical(z, dim=-1)
    interior = slice(2, -2)
    rel_gap = rms(fd[:, interior] - analytic[:, interior]) / rms(analytic[:, interior])
    assert rel_gap < 1e-3, f"dt_physical implementation diverges from a known derivative: {rel_gap}"


def test_operator_sanity(diag_val_data, diag_out_dir):
    """P4b: cross-check `dt_physical(s_true)` (per-sample central difference) against the
    balloon module's own ds/dt formula evaluated on the same ground truth. This run's real
    kappa=1.92 gives the s-equation a ~1/kappa=0.52-sample decay timescale -- faster than
    the dt=1.0 recording resolution -- so central differencing on the recorded samples is
    expected to under-resolve it and show a real, nonzero gap; that is a property of this
    physical regime's sampling, not an operator bug (P4a already rules that out). Only a
    loose sanity bound here, not a tight one; separately records the model's x_hat-to-
    x_true unit scale via least squares -- NOT assumed to be 1 (spec Section 2: "do not
    assume 1").
    """
    d = diag_val_data
    x_true, s_true, f_true, v_true, q_true = (
        _flat(d["x_true"]),
        _flat(d["s_true"]),
        _flat(d["f_true"]),
        _flat(d["v_true"]),
        _flat(d["q_true"]),
    )
    x_hat = _flat(d["x_hat"])
    haemo = d["haemo"]

    constants = HaemodynamicConstants(
        kappa=haemo["kappa"], gamma=haemo["gamma"], alpha=haemo["alpha"], E0=d["E0"], V0=d["V0"]
    )
    layer = CortexLayer(
        depth=0,
        tau=haemo["tau"],
        state=HaemodynamicState(x=x_true, s=s_true, f=f_true, v=v_true, q=q_true),
        lambda_d=0.0,
        drain_from=None,
    )
    ds_dt_formula = balloon_derivatives(layer, constants, d["order"])["ds_dt"]
    ds_dt_fd = dt_physical(s_true, dim=-1)

    # torch.gradient's boundary stencil is first-order (vs second-order interior), so
    # exclude the two boundary samples from the comparison.
    interior = slice(1, -1)
    rel_gap = rms(ds_dt_fd[:, interior] - ds_dt_formula[:, interior]) / (
        rms(ds_dt_formula[:, interior]) + 1e-8
    )

    scale = (x_hat * x_true).sum() / (x_true**2).sum().clamp(min=1e-8)

    report = {
        "fd_vs_balloon_formula_rel_gap": rel_gap,
        "kappa": haemo["kappa"],
        "implied_s_timescale_samples": 1.0 / haemo["kappa"],
        "x_hat_unit_scale_vs_x_true": scale.item(),
    }
    write_report(report, diag_out_dir, "P4_operator_sanity")

    assert math.isfinite(rel_gap)
    # Loose bound: catches real operator/unit bugs (wrong sign, wrong array axis, a
    # missing/extra (T-1) factor -- any of which would blow this up by >=10x) without
    # failing on the expected discretization bias documented above. For comparison,
    # dt_spectral on this same signal (Gibbs ringing off x_true's sharp jumps) gives a
    # rel_gap around 47 -- this bound is well below that and well above the ~0.1 a fully
    # resolved signal would show.
    assert rel_gap < 2.0, (
        f"dt_physical(s_true) vs the balloon module's own ds/dt formula gapped further "
        f"than the expected kappa-driven discretization bias explains -- check for a real "
        f"operator/unit bug (spec Section 8). Got rel_gap={rel_gap}"
    )
    assert math.isfinite(scale.item())
