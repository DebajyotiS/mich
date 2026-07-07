"""Follow-up to LS5's partial result (x_from_emit lags x_true by ~0.75, not 0 or 1): decompose
that 0.75 into its two possible sources, measured directly rather than inferred.

x_hat is built from Dp_s (FiLM-path partial derivative). x_from_emit is built from D_emit =
dt_physical(s_hat) (the plain numerical derivative of the model's own emitted s). LS4 showed
s_hat's VALUE is in phase with s_true to the noise floor -- but a derivative is a high-pass
operator (multiplies by frequency), so a phase deviation too small to move a value-domain
lag estimate (dominated by low-frequency, high-power content) can still be amplified in the
derivative if it's concentrated at higher frequencies. That would mean D_emit can be lagged
even though s_hat itself is clean -- not a contradiction, just two different measurements.

Two lags, measured independently:
  - lag(D_emit, D_true): is the true derivative of the emitted s already lagged relative to
    the true derivative of the reference s, even though the values (LS4) are not?
  - lag(Dp_s, D_emit): the FiLM-partial-vs-emitted-derivative gap itself (the u-blindness
    quantity) -- viable now that the units fix brought Dp_s from ~500x too small to ~21x,
    not pure noise.

If these two roughly add up to x_hat's ~1.0 (LS1/LS4), the 0.75 vs 1.0 split in LS5 is fully
accounted for by "D_emit itself carries most of the lag, Dp_s adds a smaller remainder" --
not a third, unexplained mechanism.
"""

from __future__ import annotations

import math

from common import (
    dt_physical,
    parabolic_subsample_lag,
    phase_slope_lag,
    upsampled_xcorr_lag,
    write_report,
)


def _flat(t):
    return t.reshape(-1, t.shape[-1])


def _lags(pred, true):
    parabolic = parabolic_subsample_lag(pred, true)
    upsampled = upsampled_xcorr_lag(pred, true, factor=10)
    phase = phase_slope_lag(pred, true, freq_cutoff=0.4)
    return {
        "parabolic": {"mean": parabolic.mean().item(), "std": parabolic.std().item()},
        "upsampled_10x": {"mean": upsampled.mean().item(), "std": upsampled.std().item()},
        "phase_slope": phase,
    }


def test_derivative_lag_decomposition(diag_val_data, diag_out_dir):
    d = diag_val_data
    s_hat, s_true, Dp_s = _flat(d["s_hat"]), _flat(d["s_true"]), _flat(d["Dp_s"])
    T = s_hat.shape[-1]

    D_emit = dt_physical(s_hat, dim=-1)
    D_true = dt_physical(s_true, dim=-1)
    Dp_s_phys = Dp_s / (T - 1)

    report = {
        "D_emit_vs_D_true": _lags(D_emit, D_true),
        "Dp_s_vs_D_emit": _lags(Dp_s_phys, D_emit),
        "Dp_s_rms": Dp_s_phys.pow(2).mean().sqrt().item(),
        "D_emit_rms": D_emit.pow(2).mean().sqrt().item(),
    }
    write_report(report, diag_out_dir, "LS6_derivative_lag_decomposition")

    assert math.isfinite(report["D_emit_vs_D_true"]["parabolic"]["mean"])


def test_ground_truth_fd_bias_lag(diag_val_data, diag_out_dir):
    """Pure ground-truth check, no model: does dt_physical's FD discretization bias
    (already shown by P4b to be a real value-RMS gap given kappa's ~0.52-sample decay
    timescale) also manifest as an apparent LAG when x_ref_phys = dt_physical(s_true) +
    kappa*s_true + gamma*(f_true-1) is compared against x_true by a broadband estimator?
    If so, LS5's x_from_emit ~0.75-sample lag is inherited FD-operator bias, not a
    model/u-blindness effect, and x_hat's model-attributable excess is only ~1.0-0.75."""
    d = diag_val_data
    s_true, f_true, x_true = _flat(d["s_true"]), _flat(d["f_true"]), _flat(d["x_true"])
    kappa, gamma = d["kappa"], d["gamma"]

    x_ref_phys = dt_physical(s_true, dim=-1) + kappa * s_true + gamma * (f_true - 1.0)

    report = {"x_ref_phys_vs_x_true": _lags(x_ref_phys, x_true)}
    write_report(report, diag_out_dir, "LS7_ground_truth_fd_bias_lag")

    assert math.isfinite(report["x_ref_phys_vs_x_true"]["parabolic"]["mean"])
