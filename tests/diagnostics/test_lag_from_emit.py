"""The discriminating measurement for the u-blindness/Dp_s-decoupling hypothesis (see
conversation record): s, f, v, q are all in phase to the noise floor (LS4) while x_hat
lags x_true by ~1.0 sample (LS1-LS3) through the identical path. Algebraically, if s is
in-phase and the residual x_hat ~= Dp_s + kappa*s_hat + gamma*(f_hat-1) is what actually
produces x, the only way x_hat's lag is consistent with s's clean phase is if Dp_s itself
(the FiLM-path partial derivative) is not the true derivative of the emitted s -- i.e. the
lag is inherited from Dp_s, not from x's own head/FiLM parameters.

This is tested directly (not by another elimination) by comparing lags, not values:
x_from_emit = dt_physical(s_hat) + kappa*s_hat + gamma*(f_hat-1) is "x as it would be if it
used the true numerical derivative of the emitted s instead of Dp_s." Its RMSE vs x_true
is already measured (UC1, corrected convention). This file measures its *lag* vs x_true at
sub-sample resolution, the same way x_hat's +1.0 was measured:

  - x_from_emit lag ~= 0 while x_hat lag ~= +1  => the delay is entirely the Dp_s-vs-true-
    derivative gap (u-blindness): Dp_s is decoupled from s's in-phase value because the
    FiLM path differentiates a different object than what the value readout of u emits.
  - x_from_emit also lags ~= +1  => even the true derivative of the emitted s gives a
    lagged x, which would be close to self-contradictory given s's clean value-phase, and
    would call the estimator (or the s/x comparison itself) back into question.
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


def test_lag_of_reconstructed_x_from_emit(diag_val_data, diag_out_dir):
    d = diag_val_data
    x_hat, s_hat, f_hat, x_true = (
        _flat(d["x_hat"]),
        _flat(d["s_hat"]),
        _flat(d["f_hat"]),
        _flat(d["x_true"]),
    )
    kappa, gamma = d["kappa"], d["gamma"]

    x_from_emit = dt_physical(s_hat, dim=-1) + kappa * s_hat + gamma * (f_hat - 1.0)

    def _lags(pred, true):
        parabolic = parabolic_subsample_lag(pred, true)
        upsampled = upsampled_xcorr_lag(pred, true, factor=10)
        phase = phase_slope_lag(pred, true, freq_cutoff=0.4)
        return {
            "parabolic": {"mean": parabolic.mean().item(), "std": parabolic.std().item()},
            "upsampled_10x": {"mean": upsampled.mean().item(), "std": upsampled.std().item()},
            "phase_slope": phase,
        }

    report = {
        "x_hat_vs_x_true": _lags(x_hat, x_true),
        "x_from_emit_vs_x_true": _lags(x_from_emit, x_true),
    }
    write_report(report, diag_out_dir, "LS5_x_from_emit_lag")

    assert math.isfinite(report["x_from_emit_vs_x_true"]["parabolic"]["mean"])
