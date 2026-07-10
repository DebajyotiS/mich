"""Redo of the x_hat lag decomposition using ONLY autograd-based quantities (never
dt_physical/FD), after LS5-LS7 showed the FD-based reconstruction (x_from_emit) carries
~0.75 samples of apparatus bias that has nothing to do with x_hat itself -- x_from_emit is
a diagnostic proxy, not the network's actual output, and decomposing its lag does not
decompose x_hat's lag (P1 already showed the tie between them is loose: residual rms
0.31-0.43).

x_rhs = Dp_s + kappa*s_hat + gamma*(f_hat-1) (P1's own quantity) is FD-free: Dp_s comes from
jacrev, s_hat/f_hat are raw decoded values. Comparing lags across x_hat -> x_rhs -> x_true
stays entirely inside clean, differentiation-of-FD-free measurements:

  - lag(x_rhs, x_true): is the network's own (autograd) training target already lagged
    relative to ground truth? This is the u-blindness mechanism's actual footprint, measured
    without going anywhere near dt_physical.
  - lag(x_hat, x_rhs): does x_hat's own phase match its (possibly-lagged) training target,
    or does x's own head/parameterization add an independent lag on top?

If lag(x_rhs, x_true) ~= lag(x_hat, x_true) ~= 1.0, x_hat has essentially fully inherited
whatever phase Dp_s carries, and the mechanism is entirely upstream (in Dp_s / u-blindness).
If lag(x_rhs, x_true) is much smaller than 1.0, x's own parameterization/training is adding
lag independently of the physics-residual target, and u-blindness alone doesn't explain it.
"""

from __future__ import annotations

import math

from common import parabolic_subsample_lag, phase_slope_lag, upsampled_xcorr_lag, write_report


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


def test_fd_free_lag_decomposition(diag_val_data, diag_out_dir):
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

    report = {
        "x_hat_vs_x_true": _lags(x_hat, x_true),
        "x_rhs_vs_x_true": _lags(x_rhs, x_true),
        "x_hat_vs_x_rhs": _lags(x_hat, x_rhs),
    }
    write_report(report, diag_out_dir, "LS9_fd_free_lag_decomposition")

    for pair in report.values():
        assert math.isfinite(pair["parabolic"]["mean"])
