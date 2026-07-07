"""Group A -- attribution. The centerpiece: how much of x's error does each mechanism
(E_op, E_deriv, E_value) own, and what does removing E_op alone (Fix 1's effect) buy?"""

from __future__ import annotations

import math

import torch
from common import (
    detect_edges,
    dt_index,
    edge_window_mask,
    rms,
    write_report,
    xcorr_lag,
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


def test_error_budget(diag_val_data, diag_out_dir):
    """A1: E_op = D_partial - D_emit, E_deriv = D_emit - D_true, E_value as in P3. Report
    each as RMS (overall + edge-window) and as a fraction of Var(x_hat - x_ref); assert
    the reconstruction identity holds (that's checkable regardless of the actual split)."""
    d = diag_val_data
    x_hat, Dp_s, s_hat, s_true, f_hat, f_true, x_true = (
        _flat(d["x_hat"]),
        _flat(d["Dp_s"]),
        _flat(d["s_hat"]),
        _flat(d["s_true"]),
        _flat(d["f_hat"]),
        _flat(d["f_true"]),
        _flat(d["x_true"]),
    )
    kappa, gamma = d["kappa"], d["gamma"]

    D_emit = dt_index(s_hat, dim=-1)
    D_true = dt_index(s_true, dim=-1)

    E_op = Dp_s - D_emit
    E_deriv = D_emit - D_true
    E_value = kappa * (s_hat - s_true) + gamma * (f_hat - f_true)

    x_ref = D_true + kappa * s_true + gamma * (f_true - 1.0)
    lhs = x_hat - x_ref
    rhs = E_op + E_deriv + E_value
    consistency_resid = lhs - rhs

    total_var = lhs.float().var().clamp(min=1e-12)
    edge_mask = _edge_mask_batch(x_true, EDGE_WINDOW)

    def _budget(e: torch.Tensor) -> dict:
        return {
            "rms_overall": rms(e),
            "rms_edge_window": rms(e[edge_mask]),
            "rms_plateau": rms(e[~edge_mask]),
            "frac_of_var_lhs": (e.float().var() / total_var).item(),
        }

    report = {
        "E_op": _budget(E_op),
        "E_deriv": _budget(E_deriv),
        "E_value": _budget(E_value),
        "lhs_rms": rms(lhs),
        "consistency_residual_rms": rms(consistency_resid),
        "consistency_residual_frac_of_lhs": (
            consistency_resid.float().pow(2).mean() / (lhs.float().pow(2).mean() + 1e-12)
        ).item(),
    }
    write_report(report, diag_out_dir, "A1_error_budget")

    for key in ("E_op", "E_deriv", "E_value"):
        assert math.isfinite(report[key]["rms_overall"])
    # The decomposition is exact algebra (E_op+E_deriv+E_value == lhs by construction);
    # this only checks we didn't introduce a bug (wrong operator, mismatched shapes/dims).
    assert report["consistency_residual_frac_of_lhs"] < 1e-6


def test_counterfactual_drives(diag_val_data, diag_out_dir):
    """A2: x_from_emit is exactly what x becomes under Fix 1 (operator gap removed).
    If close to x_true, Fix 1 alone solves it; if it still rings, the residual is the
    bandwidth term and Fix 2 is also needed."""
    d = diag_val_data
    x_hat, s_hat, f_hat, x_true = (
        _flat(d["x_hat"]),
        _flat(d["s_hat"]),
        _flat(d["f_hat"]),
        _flat(d["x_true"]),
    )
    kappa, gamma = d["kappa"], d["gamma"]

    x_from_emit = dt_index(s_hat, dim=-1) + kappa * s_hat + gamma * (f_hat - 1.0)

    def _compare(pred: torch.Tensor, true: torch.Tensor) -> dict:
        lags = xcorr_lag(pred, true)
        return {
            "rmse": rms(pred - true),
            "lag_mean_samples": lags.mean().item(),
            "lag_std_samples": lags.std().item(),
        }

    report = {
        "x_hat_vs_x_true": _compare(x_hat, x_true),
        "x_from_emit_vs_x_true": _compare(x_from_emit, x_true),
        "x_hat_vs_x_from_emit": _compare(x_hat, x_from_emit),
    }
    write_report(report, diag_out_dir, "A2_counterfactual_drives")

    for entry in report.values():
        assert math.isfinite(entry["rmse"])
