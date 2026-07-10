"""Discriminating test between the two remaining live lag-origin hypotheses (see
conversation record): is the ~1-sample x_hat-vs-x_true delay specific to x's own
production/readout, or a shared time-axis offset (model t-grid vs recorded grid) that
would show up in every signal and just wasn't looked for outside of x?

Every prior lag measurement in this suite touched at most x (LS1-LS3, full sub-sample
trio) or s (P2, integer xcorr_lag only). f, v, q have never been lag-tested at all, at any
resolution. This file runs the identical sub-sample estimator trio (parabolic, band-limited
upsampling, cross-spectrum phase slope) on all five signals side by side, through the same
metric path, so the split is read off the data rather than assumed:

  - x uniquely lagged, s/f/v/q all ~0  => offset is specific to x's own FiLM/head/readout;
    the shared temporal-mixing backbone and decoder are exonerated (s/f/v/q share that
    machinery with x and are clean).
  - all five lagged by ~the same amount => shared t-grid-to-recorded-grid offset, affecting
    every signal equally; it was invisible on s/f/v/q only because nobody had measured it
    there before now.
"""

from __future__ import annotations

import math

from common import parabolic_subsample_lag, phase_slope_lag, upsampled_xcorr_lag, write_report

SIGNALS = ("x", "s", "f", "v", "q")


def _flat(t):
    return t.reshape(-1, t.shape[-1])


def test_subsample_lag_all_signals(diag_val_data, diag_out_dir):
    d = diag_val_data
    report = {}
    for sig in SIGNALS:
        hat, true = _flat(d[f"{sig}_hat"]), _flat(d[f"{sig}_true"])
        parabolic = parabolic_subsample_lag(hat, true)
        upsampled = upsampled_xcorr_lag(hat, true, factor=10)
        phase = phase_slope_lag(hat, true, freq_cutoff=0.4)
        report[sig] = {
            "parabolic": {"mean": parabolic.mean().item(), "std": parabolic.std().item()},
            "upsampled_10x": {"mean": upsampled.mean().item(), "std": upsampled.std().item()},
            "phase_slope": phase,
        }
    write_report(report, diag_out_dir, "LS4_all_signals_subsample_lag")

    for sig in SIGNALS:
        assert math.isfinite(report[sig]["parabolic"]["mean"])
        assert math.isfinite(report[sig]["phase_slope"]["delay"])
