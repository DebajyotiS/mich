"""Group L -- localization and characterization of the delay (E_op) and ringing/gain
(E_deriv) mechanisms identified in Group A."""

from __future__ import annotations

import math

import torch
from common import (
    detect_blocks,
    detect_edges,
    dt_index,
    local_lag_at_edge,
    power_spectrum,
    rms,
    save_hist,
    save_psd_plot,
    save_scatter,
    write_report,
    xcorr_lag,
)

EDGE_WINDOW = 3
EDGE_TOL = 0.05
L5_SEGMENT = 10  # half-width of the edge-centered segment used for the edge-focused PSD


def _flat(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, t.shape[-1])


def test_delay_offsets(diag_val_data, diag_out_dir):
    """L1: a clean nonzero lag(D_partial, D_emit) matching lag(x_hat, x_true) is the
    signature of an operator delay; lag(D_emit, D_true) ~ 0 confirms the emitted-s
    derivative is itself on time."""
    d = diag_val_data
    x_hat, x_true, Dp_s, s_hat, s_true = (
        _flat(d["x_hat"]),
        _flat(d["x_true"]),
        _flat(d["Dp_s"]),
        _flat(d["s_hat"]),
        _flat(d["s_true"]),
    )
    D_emit = dt_index(s_hat, dim=-1)
    D_true = dt_index(s_true, dim=-1)

    def _summary(pred, true):
        lags = xcorr_lag(pred, true)
        return {"mean": lags.mean().item(), "std": lags.std().item()}

    report = {
        "lag_x_hat_vs_x_true": _summary(x_hat, x_true),
        "lag_Dp_s_vs_D_emit": _summary(Dp_s, D_emit),
        "lag_Dp_s_vs_D_true": _summary(Dp_s, D_true),
        "lag_D_emit_vs_D_true": _summary(D_emit, D_true),
    }
    write_report(report, diag_out_dir, "L1_delay_offsets")

    for entry in report.values():
        assert math.isfinite(entry["mean"])


def test_edge_asymmetry(diag_val_data, diag_out_dir):
    """L2: per-edge local lag(x_hat, x_true), split rising vs falling. A symmetric delay
    is zero-phase-ish; asymmetric onset-vs-offset timing points at a causal/directional
    component (e.g. the non-causal-padded but still directional TCN)."""
    d = diag_val_data
    x_hat, x_true = _flat(d["x_hat"]), _flat(d["x_true"])
    edges = detect_edges(x_true, tol=EDGE_TOL)

    rising_lags, falling_lags = [], []
    window = 8
    for i, e in enumerate(edges):
        for edge in e["rising"].tolist():
            lag = local_lag_at_edge(x_hat[i], x_true[i], edge, window)
            if lag is not None:
                rising_lags.append(lag)
        for edge in e["falling"].tolist():
            lag = local_lag_at_edge(x_hat[i], x_true[i], edge, window)
            if lag is not None:
                falling_lags.append(lag)

    rising_t = torch.tensor(rising_lags) if rising_lags else torch.tensor([float("nan")])
    falling_t = torch.tensor(falling_lags) if falling_lags else torch.tensor([float("nan")])

    report = {
        "n_rising_edges": len(rising_lags),
        "n_falling_edges": len(falling_lags),
        "rising_lag_mean": rising_t.mean().item(),
        "rising_lag_std": rising_t.std().item(),
        "falling_lag_mean": falling_t.mean().item(),
        "falling_lag_std": falling_t.std().item(),
    }
    write_report(report, diag_out_dir, "L2_edge_asymmetry")
    if rising_lags and falling_lags:
        save_hist(
            rising_lags,
            diag_out_dir / "L2_rising_lag_hist.png",
            title="Rising-edge local lag (x_hat vs x_true)",
            xlabel="lag (samples)",
        )
        save_hist(
            falling_lags,
            diag_out_dir / "L2_falling_lag_hist.png",
            title="Falling-edge local lag (x_hat vs x_true)",
            xlabel="lag (samples)",
        )

    assert report["n_rising_edges"] > 0 or report["n_falling_edges"] > 0


def test_edge_ringing(diag_val_data, diag_out_dir):
    """L3: after removing L1's global offset, peak overshoot above the true plateau
    within +-window of each rising edge, normalized by block height."""
    d = diag_val_data
    x_hat, x_true = _flat(d["x_hat"]), _flat(d["x_true"])
    N, T = x_hat.shape

    global_lag = round(xcorr_lag(x_hat, x_true).mean().item())
    x_hat_aligned = torch.roll(x_hat, shifts=-global_lag, dims=-1) if global_lag != 0 else x_hat

    edges = detect_edges(x_true, tol=EDGE_TOL)
    overshoots = []
    for i, e in enumerate(edges):
        for edge in e["rising"].tolist():
            lo, hi = edge - EDGE_WINDOW, edge + EDGE_WINDOW + 1
            if lo < 1 or hi > T:
                continue
            pre_level = x_true[i, edge - 1].item()
            post_level = x_true[i, edge].item()
            height = abs(post_level - pre_level)
            if height < 1e-6:
                continue
            peak = x_hat_aligned[i, lo:hi].max().item()
            overshoots.append((peak - post_level) / height)

    report = {
        "global_lag_removed_samples": global_lag,
        "n_edges": len(overshoots),
        "overshoot_mean": float(torch.tensor(overshoots).mean()) if overshoots else math.nan,
        "overshoot_std": float(torch.tensor(overshoots).std()) if overshoots else math.nan,
        "overshoot_p90": float(torch.tensor(overshoots).quantile(0.9)) if overshoots else math.nan,
    }
    write_report(report, diag_out_dir, "L3_edge_ringing")
    if overshoots:
        save_hist(
            overshoots,
            diag_out_dir / "L3_overshoot_hist.png",
            title="Rising-edge overshoot / block height (post global-lag removal)",
            xlabel="normalized overshoot",
        )

    assert len(overshoots) > 0


def test_gain_vs_blockwidth(diag_val_data, diag_out_dir):
    """L4: per-block plateau-mean gain error (after removing the global P4 unit-scale
    offset) vs block width. Bandwidth predicts |gain error| growing as blocks narrow."""
    d = diag_val_data
    x_hat, x_true = _flat(d["x_hat"]), _flat(d["x_true"])
    N, T = x_hat.shape

    # Same global least-squares scale as P4 -- removes the constant amplitude offset so
    # this test isolates the WIDTH-DEPENDENT component, not the overall scale mismatch.
    scale = (x_hat * x_true).sum() / (x_true**2).sum().clamp(min=1e-8)
    x_hat_corrected = x_hat / scale.clamp(min=1e-8)

    edges = detect_edges(x_true, tol=EDGE_TOL)
    widths, gains = [], []
    for i, e in enumerate(edges):
        blocks = detect_blocks(e["rising"], e["falling"])
        for start, end in blocks:
            lo, hi = start + EDGE_WINDOW, end - EDGE_WINDOW
            if hi <= lo:
                continue
            true_level = x_true[i, lo:hi].mean().item()
            pred_level = x_hat_corrected[i, lo:hi].mean().item()
            widths.append(end - start)
            gains.append((pred_level - true_level) / (abs(true_level) + 1e-3))

    report = {
        "n_blocks": len(widths),
        "gain_error_mean": float(torch.tensor(gains).mean()) if gains else math.nan,
        "gain_error_std": float(torch.tensor(gains).std()) if gains else math.nan,
    }
    if widths:
        widths_t, gains_t = torch.tensor(widths, dtype=torch.float32), torch.tensor(gains)
        median_w = widths_t.median()
        narrow = gains_t[widths_t <= median_w]
        wide = gains_t[widths_t > median_w]
        report["narrow_half_median_width"] = median_w.item()
        report["narrow_half_gain_error_rms"] = rms(narrow) if narrow.numel() else math.nan
        report["wide_half_gain_error_rms"] = rms(wide) if wide.numel() else math.nan
        save_scatter(
            widths,
            gains,
            diag_out_dir / "L4_gain_vs_width.png",
            title="Plateau gain error vs block width",
            xlabel="block width (samples)",
            ylabel="relative gain error",
        )
    write_report(report, diag_out_dir, "L4_gain_vs_blockwidth")

    assert len(widths) > 0


def test_spectra(diag_val_data, diag_out_dir):
    """L5: PSD of E_deriv over the whole trace vs restricted to edge-centered segments.
    Concentrated high-frequency power in the edge-focused PSD = a bandwidth-limit
    signature localized at transitions, not a broadband error."""
    d = diag_val_data
    s_hat, s_true, x_true = _flat(d["s_hat"]), _flat(d["s_true"]), _flat(d["x_true"])
    max_freq = d["max_freq"]
    N, T = s_hat.shape

    E_deriv = dt_index(s_hat, dim=-1) - dt_index(s_true, dim=-1)

    freqs_full, psd_full = power_spectrum(E_deriv, dim=-1)
    psd_full_mean = psd_full.mean(dim=0)
    nyquist = freqs_full.max().item()

    def _frac_above(freqs, psd, thresh):
        total = psd.sum().clamp(min=1e-12)
        above = psd[freqs > thresh].sum()
        return (above / total).item()

    edges = detect_edges(x_true, tol=EDGE_TOL)
    segments = []
    for i, e in enumerate(edges):
        for edge in torch.cat([e["rising"], e["falling"]]).tolist():
            lo, hi = edge - L5_SEGMENT, edge + L5_SEGMENT
            if lo < 0 or hi > T:
                continue
            segments.append(E_deriv[i, lo:hi])

    report = {
        "nyquist": nyquist,
        "max_freq": max_freq,
        "full_trace_frac_power_above_max_freq": _frac_above(freqs_full, psd_full_mean, max_freq),
        "full_trace_frac_power_above_90pct_nyquist": _frac_above(
            freqs_full, psd_full_mean, 0.9 * nyquist
        ),
        "n_edge_segments": len(segments),
    }
    if segments:
        seg_stack = torch.stack(segments)
        freqs_edge, psd_edge = power_spectrum(seg_stack, dim=-1)
        psd_edge_mean = psd_edge.mean(dim=0)
        report["edge_segment_frac_power_above_max_freq"] = _frac_above(
            freqs_edge, psd_edge_mean, max_freq
        )
        save_psd_plot(
            freqs_edge,
            {"edge-centered segments": psd_edge_mean.numpy()},
            diag_out_dir / "L5_psd_edge.png",
            title="PSD of E_deriv, edge-centered segments",
            vlines=[max_freq],
        )
    save_psd_plot(
        freqs_full,
        {"full trace": psd_full_mean.numpy()},
        diag_out_dir / "L5_psd_full.png",
        title="PSD of E_deriv, full trace",
        vlines=[max_freq],
    )
    write_report(report, diag_out_dir, "L5_spectra")

    assert math.isfinite(report["full_trace_frac_power_above_max_freq"])
