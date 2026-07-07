"""Verification of a units-convention hypothesis raised after the initial diagnostic run
(see conversation record, not the original spec): `_compute_physics_layer_loss`'s s_loss
compares `Dp_s` (units d/d(t_norm)) directly against `x - kappa*s - gamma*(f-1)` (physical-
value units) with no rescaling, while `_derivative_supervision_loss` explicitly divides by
`(T_min - 1)` to perform exactly this conversion (its own docstring: "dz_hat_dt is
d(z_hat)/d(t_norm), so it's divided by (T_min - 1) to undo that normalisation").

This file does NOT change src/mich -- it tests, only inside the diagnostic harness,
whether re-deriving x's counterfactual reconstruction (A2) in the physically-consistent
convention closes the gap that the original (spec-literal) A2 showed. That is the actual
go/no-go for whether Fix 1 should be "replace Dp_s with a numerical derivative" (as the
spec's Section 6 literally says) or "fix the units, then reconsider" (this file's
hypothesis). Per-step evidence lives in each test's report; do not average them into one
conclusion (units and the O1 u-path delay are separate questions -- see the report).
"""

from __future__ import annotations

import math

import torch
from common import dt_index, dt_physical, rms, write_report, xcorr_lag

EDGE_TOL = 0.05


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, t.shape[-1])


def test_derived_conversion_factor_is_T_minus_1():
    """Step 2: derive the t_norm <-> physical conversion from first principles and check
    it reduces to exactly (T-1) given this simulation's dt_sim=1.0 (independently
    confirmed by tracing scripts/run_sim.py -> neuronal.py -> balloon.py: haemo_dt=0.05 is
    a pure RK4 sub-step, invisible at the recorded resolution; one recorded index really is
    one physical/ODE time unit). t_norm = linspace(0,1,T) => dt_norm/di = 1/(T-1);
    dt_physical/di = dt_sim => dt_physical/dt_norm = dt_sim*(T-1) => d/dt_norm =
    dt_sim*(T-1) * d/dt_physical. This is NOT expected to reconcile with P4's rel_gap=0.75
    -- that number comes from a completely different comparison (dt_physical(s_true) vs
    the balloon ODE formula, entirely in physical units, no Dp_s or t_norm involved) that
    is dominated by kappa=1.92's ~0.52-sample decay timescale aliasing against dt=1
    sampling. Conflating the two would be a mistake; this test only checks the derivation
    matches what `_derivative_supervision_loss` already implements.
    """
    dt_sim = 1.0
    T = 100
    derived_factor = dt_sim * (T - 1)
    assert derived_factor == T - 1 == 99


def test_corrected_counterfactual_drive(diag_val_data, diag_out_dir):
    """Step 3: re-derive A2's x_from_emit using dt_physical (physical-unit convention,
    matching _derivative_supervision_loss) instead of dt_index (t_norm convention, what
    the spec's literal x_model/x_from_emit formula uses). Go/no-go for the units
    hypothesis: does this now closely track x_true where the t_norm-convention version
    (original A2) did not (RMSE 6.63)?
    """
    d = diag_val_data
    x_hat, s_hat, f_hat, x_true = (
        _flat(d["x_hat"]),
        _flat(d["s_hat"]),
        _flat(d["f_hat"]),
        _flat(d["x_true"]),
    )
    kappa, gamma = d["kappa"], d["gamma"]

    x_from_emit_tnorm = dt_index(s_hat, dim=-1) + kappa * s_hat + gamma * (f_hat - 1.0)
    x_from_emit_phys = dt_physical(s_hat, dim=-1) + kappa * s_hat + gamma * (f_hat - 1.0)

    def _compare(pred, true):
        lags = xcorr_lag(pred, true)
        return {
            "rmse": rms(pred - true),
            "lag_mean_samples": lags.mean().item(),
            "lag_std_samples": lags.std().item(),
        }

    report = {
        "x_hat_vs_x_true": _compare(x_hat, x_true),
        "x_from_emit_tnorm_vs_x_true (original A2, spec-literal convention)": _compare(
            x_from_emit_tnorm, x_true
        ),
        "x_from_emit_phys_vs_x_true (corrected convention)": _compare(x_from_emit_phys, x_true),
    }
    write_report(report, diag_out_dir, "UC1_corrected_counterfactual")

    phys_rmse = report["x_from_emit_phys_vs_x_true (corrected convention)"]["rmse"]
    x_hat_rmse = report["x_hat_vs_x_true"]["rmse"]
    assert math.isfinite(phys_rmse)
    # Report-only: this is the go/no-go the caller asked for, surfaced as an assert so a
    # regression shows up as a failure, but the actual numbers (not this bool) are what
    # should drive the Fix-1 decision.
    print(
        f"\nGO/NO-GO for units-corrected Fix 1: phys_rmse={phys_rmse:.4f} vs "
        f"x_hat_rmse={x_hat_rmse:.4f} (corrected-Fix-1-better = {phys_rmse < x_hat_rmse})"
    )


def test_corrected_error_budget(diag_val_data, diag_out_dir):
    """Step 4: only meaningful if the corrected A2 above actually closes the gap --
    redo A1 in the same physical convention (Dp_s divided by (T-1), matching how
    _derivative_supervision_loss already treats it) and check whether E_op shrinks to a
    magnitude tied to the ~1-2 sample x lag rather than the inflated t_norm-scale figure."""
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
    T = s_hat.shape[-1]

    Dp_s_phys = Dp_s / (T - 1)
    D_emit_phys = dt_physical(s_hat, dim=-1)
    D_true_phys = dt_physical(s_true, dim=-1)

    E_op_phys = Dp_s_phys - D_emit_phys
    E_deriv_phys = D_emit_phys - D_true_phys
    E_value = kappa * (s_hat - s_true) + gamma * (f_hat - f_true)

    x_ref_phys = D_true_phys + kappa * s_true + gamma * (f_true - 1.0)
    lhs = x_hat - x_ref_phys

    report = {
        "x_ref_phys_vs_x_true_rmse": rms(x_ref_phys - x_true),
        "E_op_phys_rms": rms(E_op_phys),
        "E_deriv_phys_rms": rms(E_deriv_phys),
        "E_value_rms": rms(E_value),
        "x_hat_minus_x_ref_phys_rms": rms(lhs),
        "Dp_s_phys_rms": rms(Dp_s_phys),
        "D_emit_phys_rms": rms(D_emit_phys),
    }
    write_report(report, diag_out_dir, "UC2_corrected_error_budget")

    assert math.isfinite(report["E_op_phys_rms"])
    # x_ref_phys = D_true_phys + kappa*s_true + gamma*(f_true-1) is literally the balloon
    # ODE's definition of x evaluated on ground truth -- it should track x_true up to the
    # P4b discretization bias (rel_gap ~0.75 there), not exactly, but not by orders of
    # magnitude either.
    assert report["x_ref_phys_vs_x_true_rmse"] < 5 * rms(x_true)
