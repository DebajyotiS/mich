"""Is Dp_s's ~1.0-sample lag structural (u-blindness) or inherited from its own training
target?

`_derivative_supervision_loss` (mich_losses.py) trains Dp_s (rescaled to physical units,
`pred / (T_min-1)`) via MSE directly against `torch.gradient(true_s, dim=2)` -- the exact FD
operator LS7/P4b showed carries ~0.75 samples of apparent lag against kappa's fast decay.
That's the one place FD enters training at all, and it's a supervision target for the exact
quantity now indicted for x_hat's lag. Two distinct mechanisms, two different fixes:

  - If Dp_s tracks its own FD target tightly (lag(Dp_s, FD_target) ~= 0) AND that target is
    itself lagged relative to the true ds/dt (lag(FD_target, true_ds_dt) ~= lag(Dp_s,
    true_ds_dt)), then Dp_s's lag is inherited from a biased supervision target -- cheap fix:
    change the target (supervise against the analytic balloon derivative, or drop the term).
  - If Dp_s is lagged even relative to its own FD target (lag(Dp_s, FD_target) is itself
    sizable), that's the structural u-blindness mechanism, and static-u is the answer.

true_ds_dt = x_true - kappa*s_true - gamma*(f_true-1) is used directly, with NO
differentiation anywhere: LS8 already confirmed (via RK4 re-simulation, zero FD) that
x_true/s_true/f_true are correctly time-aligned at matched indices, so this algebraic
combination *is* the exact ds_true/dt, not an approximation.
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


def test_dp_s_vs_its_own_supervision_target(diag_val_data, diag_out_dir):
    d = diag_val_data
    Dp_s, s_true, f_true, x_true = (
        _flat(d["Dp_s"]),
        _flat(d["s_true"]),
        _flat(d["f_true"]),
        _flat(d["x_true"]),
    )
    kappa, gamma = d["kappa"], d["gamma"]
    T = Dp_s.shape[-1]

    Dp_s_phys = Dp_s / (T - 1)
    fd_target = dt_physical(
        s_true, dim=-1
    )  # exactly what _derivative_supervision_loss trains against
    true_ds_dt = x_true - kappa * s_true - gamma * (f_true - 1.0)  # FD-free, exact (see LS8)

    report = {
        "Dp_s_rms": Dp_s_phys.pow(2).mean().sqrt().item(),
        "fd_target_rms": fd_target.pow(2).mean().sqrt().item(),
        "true_ds_dt_rms": true_ds_dt.pow(2).mean().sqrt().item(),
        "Dp_s_vs_fd_target": _lags(Dp_s_phys, fd_target),
        "fd_target_vs_true_ds_dt": _lags(fd_target, true_ds_dt),
        "Dp_s_vs_true_ds_dt": _lags(Dp_s_phys, true_ds_dt),
    }
    write_report(report, diag_out_dir, "LS10_dsdt_supervision_target_check")

    for key in ("Dp_s_vs_fd_target", "fd_target_vs_true_ds_dt", "Dp_s_vs_true_ds_dt"):
        assert math.isfinite(report[key]["parabolic"]["mean"])
