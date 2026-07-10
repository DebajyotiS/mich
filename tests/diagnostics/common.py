"""Shared numerics for the x-error attribution diagnostics.

One canonical derivative operator (`dt_index`) is used everywhere the spec's `Dt(.)`
appears, so `E_deriv = dt_index(s_hat - s_true) = dt_index(s_hat) - dt_index(s_true)`
by linearity -- the same operator for both terms is what makes that identity hold.
`dt_spectral` is a second, independent derivative estimate used only as a cross-check
(see spec Section 2 "implement a windowed spectral derivative and report both"), not
part of the A1 error-budget formula itself.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import open_dict
from scipy.signal.windows import tukey


def dt_index(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Central-difference d/d(t_norm), scaled by (T-1) to undo the [0,1] time-grid
    normalisation -- matches the units of the network's analytic `dz_hat_dt`."""
    T = z.shape[dim]
    return torch.gradient(z, dim=dim)[0] * (T - 1)


def dt_physical(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Raw per-index central difference, no (T-1) rescaling.

    Ground-truth latents (s_true, f_true, ...) are sampled at physical dt=1.0 (one
    simulation time unit per index), so this IS ds_true/dt_physical directly -- the same
    scale as the balloon ODE's RHS (`x - kappa*s - gamma*(f-1)`). This is deliberately a
    different operator from `dt_index`: `dt_index` produces d/d(t_norm) to match the
    network's analytic `dz_hat_dt` (which differentiates w.r.t. the [0,1]-normalised time
    grid the decoder is conditioned on, and is `(T-1)x` larger for the same trajectory --
    see `_derivative_supervision_loss`'s `pred_src / (T_min - 1)` rescaling, which performs
    the same t_norm -> physical conversion in the other direction). Use `dt_physical` only
    when comparing against an explicitly physical-scale quantity (e.g. `balloon_derivatives`
    in P4); use `dt_index` everywhere a comparison against `Dp_*` (`dz_hat_dt`) is involved.
    """
    return torch.gradient(z, dim=dim)[0]


def dt_spectral(z: torch.Tensor, dim: int = -1, taper: float = 0.1) -> torch.Tensor:
    """Windowed spectral derivative: Tukey-taper the ends (non-periodic boundary),
    differentiate in the frequency domain, transform back. Not used to reconstruct the
    signal (no un-windowing) -- only a cross-check away from the tapered boundary."""
    z = z.movedim(dim, -1)
    T = z.shape[-1]
    window = torch.as_tensor(tukey(T, alpha=taper), dtype=z.dtype, device=z.device)
    zw = z * window
    freqs = torch.fft.rfftfreq(T, d=1.0 / (T - 1)).to(device=z.device, dtype=z.dtype)
    Z = torch.fft.rfft(zw, dim=-1)
    dZ = Z * (2j * torch.pi * freqs)
    dz = torch.fft.irfft(dZ, n=T, dim=-1)
    return dz.movedim(-1, dim)


def xcorr_lag(pred: torch.Tensor, true: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Peak zero-padded FFT cross-correlation lag (samples), batched over leading dims.
    Positive lag = `pred` is delayed relative to `true` (pred(t) ~ true(t - lag)).

    CORRECTED SIGN: the naive `irfft(rfft(true)*rfft(pred).conj())` construction (the
    form used by `MICHLoggingMixin._neural_recovery_metrics`, which this was originally
    copied from) actually places its peak at -delay, not +delay -- verified empirically
    with an unambiguous integer `torch.roll` (no interpolation involved): `pred =
    roll(true, +3)` (pred genuinely delayed by 3 samples) made the naive formula return
    -3.0, the opposite of its own docstring's claimed convention. The `-` below is that
    correction; see test_lag_subsample.py::test_sign_convention_and_subsample_accuracy_self_check
    for the check that would catch a regression here. NOTE: this means every lag number
    reported by earlier diagnostics in this suite (L1/L2/L3/A2/O1/O2), which called the
    uncorrected version, has the opposite sign from how it was originally interpreted.
    """
    pred = pred.movedim(dim, -1).float()
    true = true.movedim(dim, -1).float()
    T = pred.shape[-1]
    xcorr = torch.fft.irfft(
        torch.fft.rfft(true, n=2 * T) * torch.fft.rfft(pred, n=2 * T).conj(), n=2 * T
    )
    lags = torch.fft.fftfreq(2 * T, d=1.0 / (2 * T)).long().to(xcorr.device)
    return -lags[xcorr.argmax(dim=-1)].float()


def local_lag_at_edge(
    pred_row: torch.Tensor, true_row: torch.Tensor, edge: int, window: int
) -> float | None:
    """xcorr_lag restricted to a +-window slice around a single edge index. None if the
    window would run off either end of the trace."""
    T = pred_row.shape[0]
    lo, hi = edge - window, edge + window
    if lo < 0 or hi > T:
        return None
    return xcorr_lag(pred_row[lo:hi], true_row[lo:hi]).item()


def xcorr_curve(pred: torch.Tensor, true: torch.Tensor, dim: int = -1):
    """Same zero-padded FFT cross-correlation as `xcorr_lag`, but returns the full curve
    (not just the argmax), reordered to strictly increasing (corrected-sign, see
    `xcorr_lag`) lag so neighbouring array indices are always neighbouring integer lags
    (needed for parabolic interpolation -- without reordering, the FFT's [0..T-1, -T..-1]
    wraparound would occasionally put the "neighbour" of the peak at the wrong end of the
    array). Returns (xcorr[..., 2T], lags[2T])."""
    pred = pred.movedim(dim, -1).float()
    true = true.movedim(dim, -1).float()
    T = pred.shape[-1]
    n = 2 * T
    xcorr = torch.fft.irfft(torch.fft.rfft(true, n=n) * torch.fft.rfft(pred, n=n).conj(), n=n)
    lags = -torch.fft.fftfreq(n, d=1.0 / n).long().to(xcorr.device)  # sign-corrected, see xcorr_lag
    order = torch.argsort(lags)
    return xcorr[..., order], lags[order]


def parabolic_subsample_lag(pred: torch.Tensor, true: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sub-sample lag via parabolic interpolation of the 3 xcorr values around the integer
    peak: delta = 0.5*(c[-1]-c[+1]) / (c[-1] - 2*c[0] + c[+1]), refined_lag = argmax + delta.
    Same sign convention as `xcorr_lag` (positive = pred delayed vs true). Batched over
    leading dims; `delta` is clamped to +-1 sample since the interpolation is only valid
    within one step of the discrete peak."""
    xcorr, lags = xcorr_curve(pred, true, dim=dim)
    n = xcorr.shape[-1]
    peak_idx = xcorr.argmax(dim=-1)
    left_idx = (peak_idx - 1).clamp(min=0)
    right_idx = (peak_idx + 1).clamp(max=n - 1)

    c0 = torch.gather(xcorr, -1, peak_idx.unsqueeze(-1)).squeeze(-1)
    cm1 = torch.gather(xcorr, -1, left_idx.unsqueeze(-1)).squeeze(-1)
    cp1 = torch.gather(xcorr, -1, right_idx.unsqueeze(-1)).squeeze(-1)

    denom = cm1 - 2 * c0 + cp1
    delta = torch.where(denom.abs() > 1e-12, 0.5 * (cm1 - cp1) / denom, torch.zeros_like(denom))
    delta = delta.clamp(-1.0, 1.0)
    return lags[peak_idx].float() + delta


def parabolic_lag_at_edge(
    pred_row: torch.Tensor, true_row: torch.Tensor, edge: int, window: int
) -> float | None:
    """Sub-sample analogue of `local_lag_at_edge`, for the per-block spread check: a real
    systematic delay should be consistent edge to edge; an estimator-quantization floor
    artifact should scatter."""
    T = pred_row.shape[0]
    lo, hi = edge - window, edge + window
    if lo < 0 or hi > T:
        return None
    return parabolic_subsample_lag(pred_row[lo:hi], true_row[lo:hi]).item()


def upsampled_xcorr_lag(
    pred: torch.Tensor, true: torch.Tensor, factor: int = 10, dim: int = -1
) -> torch.Tensor:
    """Sub-sample lag via band-limited resampling (scipy.signal.resample, correct Nyquist-
    bin handling for real signals -- deliberately not a hand-rolled FFT zero-pad) followed
    by the same integer-argmax xcorr, then dividing by `factor`. Independent of the
    parabolic method's local-3-point assumption, so agreement between the two is
    informative on its own."""
    import scipy.signal as sps

    pred = pred.movedim(dim, -1).float()
    true = true.movedim(dim, -1).float()
    orig_shape = pred.shape
    T = orig_shape[-1]

    pred_np = pred.reshape(-1, T).numpy()
    true_np = true.reshape(-1, T).numpy()
    pred_up = sps.resample(pred_np, T * factor, axis=-1)
    true_up = sps.resample(true_np, T * factor, axis=-1)

    lag_up = xcorr_lag(torch.from_numpy(pred_up), torch.from_numpy(true_up), dim=-1)
    return (lag_up / factor).reshape(orig_shape[:-1])


def phase_slope_lag(
    pred: torch.Tensor, true: torch.Tensor, dim: int = -1, freq_cutoff: float = 0.4
) -> dict[str, float]:
    """Delay from the slope of the unwrapped cross-spectrum phase vs frequency, restricted
    to (0, freq_cutoff] (default matches B1's ~99%-power bandwidth) and weighted by
    cross-spectrum magnitude. Coherently averages the cross-spectrum across leading batch
    dims first (appropriate when a SHARED delay is expected across the population, not a
    per-row one -- boosts SNR relative to per-row phase estimates).

    Convention check (derived, not assumed): if pred(t) = true(t - d) (pred delayed by d),
    then Pred(f) = True(f)*exp(-2*pi*i*f*d), so angle(True(f)*conj(Pred(f))) = 2*pi*f*d --
    matches xcorr_lag's "positive = pred delayed vs true" when delay = slope/(2*pi). This
    is verified empirically against xcorr_lag on a synthetic impulse pair in
    test_lag_subsample.py::test_sign_convention_self_check, not just asserted here.

    Returns {"delay": ..., "phase_residual_std": ...} -- a large residual means the phase
    isn't linear, i.e. this isn't a clean single delay, independent of what the slope says.
    """
    pred = pred.movedim(dim, -1).float()
    true = true.movedim(dim, -1).float()
    T = pred.shape[-1]

    Pred = torch.fft.rfft(pred, dim=-1)
    True_ = torch.fft.rfft(true, dim=-1)
    cross = True_ * Pred.conj()
    while cross.dim() > 1:
        cross = cross.mean(dim=0)

    freqs = torch.fft.rfftfreq(T, d=1.0).numpy()
    mag = cross.abs().numpy()
    phase = np.unwrap(np.angle(cross.numpy()))

    band = (freqs > 0) & (freqs <= freq_cutoff)
    f, p, w = freqs[band], phase[band], mag[band]

    A = np.stack([np.ones_like(f), f], axis=1)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(sw[:, None] * A, sw * p, rcond=None)
    a, b = coef
    residual = p - (a + b * f)
    residual_std = (
        float(np.sqrt(np.average(residual**2, weights=w))) if w.sum() > 0 else float("nan")
    )

    return {"delay": float(b / (2 * np.pi)), "phase_residual_std": residual_std}


def detect_edges(x: torch.Tensor, tol: float = 1e-3) -> list[dict[str, torch.Tensor]]:
    """x: [N, T] piecewise-constant traces. Returns one {'rising','falling'} dict per row;
    each index is the sample immediately AFTER the jump."""
    N, T = x.shape
    d = x[:, 1:] - x[:, :-1]
    out = []
    for i in range(N):
        jump = torch.nonzero(d[i].abs() > tol, as_tuple=True)[0]
        idx = jump + 1
        rising = idx[d[i, jump] > 0]
        falling = idx[d[i, jump] < 0]
        out.append({"rising": rising, "falling": falling})
    return out


def edge_window_mask(T: int, edges: torch.Tensor, window: int) -> torch.Tensor:
    """Boolean [T] mask covering +-window around every edge index in `edges`."""
    mask = torch.zeros(T, dtype=torch.bool)
    for e in edges.tolist():
        lo, hi = max(0, e - window), min(T, e + window + 1)
        mask[lo:hi] = True
    return mask


def detect_blocks(rising: torch.Tensor, falling: torch.Tensor) -> list[tuple[int, int]]:
    """Interior blocks (start, end) bounded by two consecutive edges of either kind.
    The first segment (0 -> first edge) and last (last edge -> T) are dropped since
    their true width is truncated by the trace boundary, not representative."""
    edges = torch.sort(torch.cat([rising, falling])).values.tolist()
    return list(zip(edges[:-1], edges[1:], strict=False))


def power_spectrum(x: torch.Tensor, dim: int = -1, d: float = 1.0):
    """Plain periodogram (no segment-averaging -- traces are too short for Welch)."""
    x = x.movedim(dim, -1).float()
    T = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)
    psd = (X.abs() ** 2) / T
    freqs = torch.fft.rfftfreq(T, d=d)
    return freqs, psd


def rms(x: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(x.float() ** 2)).item()


def _to_jsonable(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().float().cpu()
        return v.tolist() if v.numel() > 1 else v.item()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {k: _to_jsonable(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(vv) for vv in v]
    return v


def write_report(data: dict, out_dir: Path, name: str) -> Path:
    """Write a report dict as pretty-printed JSON, print it, and return the path."""
    path = Path(out_dir) / f"{name}.json"
    payload = {k: _to_jsonable(v) for k, v in data.items()}
    path.write_text(json.dumps(payload, indent=2))
    print(f"\n=== {name} ===")
    print(json.dumps(payload, indent=2))
    return path


def save_hist(values, path: Path, *, title: str, xlabel: str) -> Path:
    fig, ax = plt.subplots()
    ax.hist(np.asarray(values), bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def save_scatter(x, y, path: Path, *, title: str, xlabel: str, ylabel: str) -> Path:
    fig, ax = plt.subplots()
    ax.scatter(np.asarray(x), np.asarray(y), s=8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def save_psd_plot(
    freqs, psds: dict[str, np.ndarray], path: Path, *, title: str, vlines: list[float] | None = None
) -> Path:
    fig, ax = plt.subplots()
    for label, psd in psds.items():
        ax.plot(np.asarray(freqs), np.asarray(psd), label=label)
    for v in vlines or []:
        ax.axvline(v, linestyle="--", color="gray")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("frequency (cycles / unit time)")
    ax.set_ylabel("power")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def build_val_loader(full_cfg):
    """Rebuild the val split from a run's saved config, forcing float32 storage to match
    the model's fp32 params (mirrors scripts/eval_mich.py's load_test_split_batch)."""
    dm_cfg = deepcopy(full_cfg.datamodule)
    with open_dict(dm_cfg):
        dm_cfg.data.return_latents = True
        dm_cfg.data.return_meta = True
        dm_cfg.data.dtype = "float32"
    datamodule = instantiate(dm_cfg)
    datamodule.setup()
    return datamodule.val_dataloader()


def _forward_and_gather(
    model,
    batch: dict,
    device: torch.device,
    xmix_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Replays HeinzleNet.forward manually so an ablation can be spliced in between
    temporal_mixing and spatial_decoder (needed by O1), then gathers every signal at
    every valid source voxel across the full T grid. Returns [N, L, T] tensors, N = total
    valid sources in this batch."""
    import sys as _sys
    from pathlib import Path as _Path

    scripts_dir = _Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in _sys.path:
        _sys.path.insert(0, str(scripts_dir))
    from eval_mich import gather_source_traces  # noqa: E402

    bold = batch["bold"].to(device).float()
    source_position = batch["source_position"].to(device)
    num_sources = batch["num_sources"].to(device)

    with torch.no_grad():
        bold_norm = (
            model.normaliser(bold, source_position, num_sources)
            if model.normaliser is not None
            else bold
        )
        time_grid = model._make_time_grid(
            B=bold.shape[0], T=bold.shape[2], device=bold.device, dtype=bold.dtype
        )

        net = model.heinzle_net
        xmix = net.layer_mixing(bold_norm)
        xenc = net.spatial_encoder(xmix)
        u = net.temporal_mixing(xenc)
        if xmix_transform is not None:
            u = xmix_transform(u)
        manifest = net.spatial_decoder(u, time_grid, return_gradients=True)
        z_hat, dz_hat_dt = manifest.z_hat, manifest.grads

    out: dict[str, torch.Tensor] = {}
    for sig in ("x", "s", "f", "v", "q"):
        idx = model._signal_index(sig)
        out[f"{sig}_hat"] = gather_source_traces(z_hat[:, idx], source_position, num_sources)
        out[f"Dp_{sig}"] = gather_source_traces(dz_hat_dt[:, idx], source_position, num_sources)
    for sig, key in (("s", "s"), ("f", "f"), ("v", "v"), ("q", "q"), ("x", "neural")):
        true_full = batch[key].to(device).float()
        out[f"{sig}_true"] = gather_source_traces(true_full, source_position, num_sources)
    return out
