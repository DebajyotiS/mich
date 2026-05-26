"""
Evaluate a trained MICH (PINN) or SupervisedMICH model on H5 dataset(s).

Usage:
    # PINN — single file:
    python scripts/eval_mich.py -c <checkpoint.ckpt> -d <data.h5>

    # PINN — full gridsearch directory:
    python scripts/eval_mich.py -c <checkpoint.ckpt> --data-dir data/gridsearch/

    # Supervised baseline:
    python scripts/eval_mich.py --model-type supervised -c <checkpoint.ckpt> --data-dir data/gridsearch/

    # Compare both in one W&B run:
    python scripts/eval_mich.py --model-type both \
        -c <pinn.ckpt> --supervised-checkpoint <sup.ckpt> --data-dir data/gridsearch/
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

import rootutils
root = rootutils.setup_root(__file__, pythonpath=True, cwd=False)

from src.models.mich import MICH
from src.models.supervised import SupervisedMICH
from src.utils.plotting import plot_latent_layers, plot_neural_bold_layers


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(data_path: str, n: int, device: torch.device) -> dict[str, torch.Tensor]:
    with h5py.File(data_path, "r") as h5f:
        bold    = torch.tensor(h5f["layer_0"]["bold"][:n], dtype=torch.float32)
        neural  = torch.tensor(h5f["layer_0"]["x"][:n],   dtype=torch.float32)
        true_s  = torch.tensor(h5f["layer_0"]["s"][:n],   dtype=torch.float32)
        true_f  = torch.tensor(h5f["layer_0"]["f"][:n],   dtype=torch.float32)
        true_v  = torch.tensor(h5f["layer_0"]["v"][:n],   dtype=torch.float32)
        true_q  = torch.tensor(h5f["layer_0"]["q"][:n],   dtype=torch.float32)
        src_pos = torch.tensor(h5f["meta"]["source_position"][:n], dtype=torch.long)

    bold    = bold.unsqueeze(1).to(device)
    neural  = neural.unsqueeze(1).to(device)
    true_s  = true_s.unsqueeze(1).to(device)
    true_f  = true_f.unsqueeze(1).to(device)
    true_v  = true_v.unsqueeze(1).to(device)
    true_q  = true_q.unsqueeze(1).to(device)
    src_pos = src_pos.to(device)
    return dict(bold=bold, neural=neural, true_s=true_s, true_f=true_f,
                true_v=true_v, true_q=true_q, src_pos=src_pos)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _resize_normaliser_buffers(model, sd):
    for buf_name in ("running_mean", "running_M2"):
        key = f"normaliser.{buf_name}"
        if key in sd:
            model.normaliser.register_buffer(buf_name, torch.zeros_like(sd[key]))


def load_mich(checkpoint_path: str, device: torch.device) -> MICH:
    raw = OmegaConf.load(root / "config" / "model" / "longfreq.yaml")
    cfg = OmegaConf.create({"model": raw})
    model: MICH = instantiate(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    _resize_normaliser_buffers(model, sd)
    model.load_state_dict(sd)
    return model.to(device).eval()


def load_supervised(checkpoint_path: str, device: torch.device) -> SupervisedMICH:
    raw = OmegaConf.load(root / "config" / "model" / "supervised.yaml")
    cfg = OmegaConf.create({"model": raw})
    model: SupervisedMICH = instantiate(cfg.model)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    _resize_normaliser_buffers(model, sd)
    model.load_state_dict(sd)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(pred_neural: torch.Tensor, true_neural: torch.Tensor) -> dict[str, float]:
    """pred/true: [N, L, T] at source voxel."""
    pred, true = pred_neural.float(), true_neural.float()
    T = pred.shape[-1]
    flat_pred = pred.reshape(-1, T)
    flat_true = true.reshape(-1, T)

    ss_res = ((flat_true - flat_pred) ** 2).sum(dim=-1)
    ss_tot = ((flat_true - flat_true.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
    r2 = (1 - ss_res / ss_tot.clamp(min=1e-8)).mean().item()

    p_c = flat_pred - flat_pred.mean(dim=-1, keepdim=True)
    t_c = flat_true - flat_true.mean(dim=-1, keepdim=True)
    pearson = (
        (p_c * t_c).sum(dim=-1)
        / (p_c.norm(dim=-1) * t_c.norm(dim=-1)).clamp(min=1e-8)
    ).mean().item()

    xcorr = torch.fft.irfft(
        torch.fft.rfft(flat_true, n=2 * T) * torch.fft.rfft(flat_pred, n=2 * T).conj(),
        n=2 * T,
    )
    lags = torch.fft.fftfreq(2 * T, d=1.0 / (2 * T)).long().to(xcorr.device)
    peak_lag = lags[xcorr.argmax(dim=-1)].float().mean().item()

    return {"r2": r2, "pearson": pearson, "lag_samples": peak_lag}


# ---------------------------------------------------------------------------
# Per-model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_mich(model: MICH, bold: torch.Tensor):
    """Returns (pred_neural [N,L,T,H,W], pred_bold [N,L,T,H,W])."""
    bold_norm = model.normaliser.normalize(bold) if model.normaliser is not None else bold
    time_grid = model._make_time_grid(B=bold.shape[0], T=bold.shape[2],
                                      device=bold.device, dtype=bold.dtype)
    z_hat = model(bold_norm, time_grid, return_gradients=False).z_hat

    pred_bold = MICH._compute_bold(
        z_hat[:, MICH._signal_index("v")],
        z_hat[:, MICH._signal_index("q")],
        acquisition=model.hparams.acquisition,
        V0=model.hparams.V0,
    )
    if model._psf is not None:
        N, L, T, H, W = pred_bold.shape
        blurred = []
        for li in range(L):
            kernel = getattr(model, f"_psf_kernel_{li}").to(pred_bold.device)
            pad = kernel.shape[-1] // 2
            x = pred_bold[:, li].reshape(N * T, 1, H, W)
            blurred.append(F.conv2d(x, kernel, padding=pad).reshape(N, T, H, W).to(pred_bold.dtype))
        pred_bold = torch.stack(blurred, dim=1)

    pred_neural = z_hat[:, MICH._signal_index("x")]  # [N, L, T, H, W]
    return pred_neural, pred_bold, z_hat


@torch.no_grad()
def infer_supervised(model: SupervisedMICH, bold: torch.Tensor):
    """Returns pred_neural [N,L,T,H,W]."""
    bold_norm = model.normaliser.normalize(bold) if model.normaliser is not None else bold
    return model.net(bold_norm)  # [N, L, T, H, W]


# ---------------------------------------------------------------------------
# Eval one file
# ---------------------------------------------------------------------------

def eval_one_mich(model: MICH, data: dict, n_samples: int, label: str, prefix: str):
    bold    = data["bold"]
    src_pos = data["src_pos"]
    pred_neural, pred_bold, z_hat = infer_mich(model, bold)

    true_bold = (model.normaliser.denormalize(model.normaliser.normalize(bold))
                 if model.normaliser is not None else bold)

    N = src_pos.shape[0]
    b_idx = torch.arange(N)
    sh = src_pos[:, 0].long()
    sw = src_pos[:, 1].long()

    pred_neural_src = pred_neural[b_idx, :, :, sh, sw]  # [N, L, T]
    true_neural_src = data["neural"][b_idx, :, :, sh, sw]
    metrics = compute_metrics(pred_neural_src, true_neural_src)

    bn_images, lat_images = [], []
    for i in range(n_samples):
        s_h, s_w = int(sh[i]), int(sw[i])
        caption = f"{label} | sample {i}"

        fig_bn = plot_neural_bold_layers(
            pred_bold=pred_bold[i, :, :, s_h, s_w].float(),
            true_bold=true_bold[i, :, :, s_h, s_w].float(),
            pred_neural=pred_neural[i, :, :, s_h, s_w].float(),
            true_neural=data["neural"][i, :, :, s_h, s_w].float(),
            source_layer=torch.zeros(1, dtype=torch.long),
            source_pos=src_pos[i:i+1],
        )
        bn_images.append(wandb.Image(fig_bn, caption=caption))
        plt.close(fig_bn)

        fig_lat = plot_latent_layers(
            pred_s=z_hat[i, MICH._signal_index("s"), :, :, s_h, s_w].float(),
            true_s=data["true_s"][i, :, :, s_h, s_w].float(),
            pred_f=z_hat[i, MICH._signal_index("f"), :, :, s_h, s_w].float(),
            true_f=data["true_f"][i, :, :, s_h, s_w].float(),
            pred_v=z_hat[i, MICH._signal_index("v"), :, :, s_h, s_w].float(),
            true_v=data["true_v"][i, :, :, s_h, s_w].float(),
            pred_q=z_hat[i, MICH._signal_index("q"), :, :, s_h, s_w].float(),
            true_q=data["true_q"][i, :, :, s_h, s_w].float(),
            title=f"{label} — latents",
        )
        lat_images.append(wandb.Image(fig_lat, caption=caption))
        plt.close(fig_lat)

    return bn_images, lat_images, {f"{prefix}/{k}": v for k, v in metrics.items()}


def eval_one_supervised(model: SupervisedMICH, data: dict, n_samples: int, label: str, prefix: str):
    bold    = data["bold"]
    src_pos = data["src_pos"]
    pred_neural = infer_supervised(model, bold)

    N = src_pos.shape[0]
    b_idx = torch.arange(N)
    sh = src_pos[:, 0].long()
    sw = src_pos[:, 1].long()

    pred_neural_src = pred_neural[b_idx, :, :, sh, sw].float()
    true_neural_src = data["neural"][b_idx, :, :, sh, sw].float()
    true_bold_src   = bold[b_idx, :, :, sh, sw].float()
    metrics = compute_metrics(pred_neural_src, true_neural_src)

    images = []
    for i in range(n_samples):
        s_h, s_w = int(sh[i]), int(sw[i])
        caption = f"{label} | sample {i}"
        fig = plot_neural_bold_layers(
            pred_bold=true_bold_src[i].float(),
            true_bold=true_bold_src[i].float(),
            pred_neural=pred_neural[i, :, :, s_h, s_w].float(),
            true_neural=data["neural"][i, :, :, s_h, s_w].float(),
            source_layer=torch.zeros(1, dtype=torch.long),
            source_pos=src_pos[i:i+1],
        )
        images.append(wandb.Image(fig, caption=caption))
        plt.close(fig)

    return images, {f"{prefix}/{k}": v for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def _parse_kappa_tau(stem: str):
    """Extract (kappa, tau) from filenames like 'kappa1.9155_tau2.6645'."""
    m = re.search(r"kappa([0-9.]+)_tau([0-9.]+)", stem)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def plot_grid_heatmaps(
    grid_metrics: dict[str, dict[str, float]],  # {stem: {"pinn/r2": v, "supervised/r2": v, ...}}
    metric: str = "r2",
) -> plt.Figure | None:
    entries = []
    for stem, vals in grid_metrics.items():
        kappa, tau = _parse_kappa_tau(stem)
        if kappa is None:
            continue
        entries.append((kappa, tau, vals))

    if not entries:
        return None

    kappas = sorted(set(e[0] for e in entries))
    taus   = sorted(set(e[1] for e in entries))
    k_idx  = {k: i for i, k in enumerate(kappas)}
    t_idx  = {t: i for i, t in enumerate(taus)}

    models = [p for p in ("pinn", "supervised") if any(f"{p}/{metric}" in v for _, _, v in entries)]
    if not models:
        return None

    # Compute shared color scale across all models
    all_vals = []
    grids = {}
    for prefix in models:
        grid = np.full((len(kappas), len(taus)), np.nan)
        for kappa, tau, vals in entries:
            v = vals.get(f"{prefix}/{metric}")
            if v is not None:
                grid[k_idx[kappa], t_idx[tau]] = v
        grids[prefix] = grid
        all_vals.extend(grid[np.isfinite(grid)].tolist())
    vmin, vmax = (min(all_vals), max(all_vals)) if all_vals else (0, 1)

    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), constrained_layout=True)
    if len(models) == 1:
        axes = [axes]

    for ax, prefix in zip(axes, models):
        grid = grids[prefix]
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       vmin=vmin, vmax=vmax, cmap="RdYlGn")
        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f"{t:.2f}" for t in taus], fontsize=8)
        ax.set_yticks(range(len(kappas)))
        ax.set_yticklabels([f"{k:.4f}" for k in kappas], fontsize=8)
        ax.set_xlabel("tau")
        ax.set_ylabel("kappa")
        ax.set_title(f"{prefix} — {metric}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for ki in range(len(kappas)):
            for ti in range(len(taus)):
                v = grid[ki, ti]
                if np.isfinite(v):
                    ax.text(ti, ki, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="black" if 0.2 < v < 0.85 else "white")

    fig.suptitle(f"Neural recovery {metric} across haemodynamic grid", fontsize=12)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MICH / SupervisedMICH on H5 dataset(s)")
    CKPT_DEFAULT = "/media/RCPNAS/Data2/korach/inversion/results_julie/mich-bold-inversion/single-layer_julie/checkpoints/last-v72.ckpt"

    parser.add_argument("--model-type", default="pinn", choices=["pinn", "supervised", "both"],
                        help="Which model(s) to evaluate")
    parser.add_argument("-c", "--checkpoint",             default=CKPT_DEFAULT,
                        help="PINN checkpoint path")
    parser.add_argument("--supervised-checkpoint",        default=None,
                        help="SupervisedMICH checkpoint path (required when --model-type supervised/both)")
    parser.add_argument("-d", "--data",                   default=None)
    parser.add_argument("--data-dir",                     default="data/gridsearch")
    parser.add_argument("-n", "--n-samples",              type=int, default=3)
    parser.add_argument("--wandb-project",                default="mich-bold-inversion")
    parser.add_argument("--wandb-entity",                 default="julie-korach-epfl")
    parser.add_argument("--wandb-run-name",               default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data:
        h5_files = [Path(args.data)]
    else:
        h5_files = sorted(Path(args.data_dir).glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No H5 files found in {args.data_dir}")

    run_name = args.wandb_run_name or f"eval/{args.model_type}"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={"model_type": args.model_type, "n_samples": args.n_samples,
                "n_files": len(h5_files)},
    )

    mich_model, sup_model = None, None
    if args.model_type in ("pinn", "both"):
        print(f"Loading PINN from {args.checkpoint}")
        mich_model = load_mich(args.checkpoint, device)
    if args.model_type in ("supervised", "both"):
        if not args.supervised_checkpoint:
            raise ValueError("--supervised-checkpoint required for supervised/both")
        print(f"Loading SupervisedMICH from {args.supervised_checkpoint}")
        sup_model = load_supervised(args.supervised_checkpoint, device)

    all_media = {}
    all_metrics = []
    grid_metrics: dict[str, dict[str, float]] = {}  # {stem: {prefix/metric: value}}

    for h5_path in h5_files:
        label = h5_path.stem
        print(f"  evaluating {label} ...")
        data = load_samples(str(h5_path), args.n_samples, device)
        grid_metrics[label] = {}

        if mich_model is not None:
            bn_imgs, lat_imgs, metrics = eval_one_mich(mich_model, data, args.n_samples, label, prefix="pinn")
            all_media.setdefault("pinn/predictions", []).extend(bn_imgs)
            all_media.setdefault("pinn/latents",     []).extend(lat_imgs)
            all_metrics.append({"file": label, **metrics})
            grid_metrics[label].update(metrics)

        if sup_model is not None:
            imgs, metrics = eval_one_supervised(sup_model, data, args.n_samples, label, prefix="supervised")
            all_media.setdefault("supervised/predictions", []).extend(imgs)
            all_metrics.append({"file": label, **metrics})
            grid_metrics[label].update(metrics)

    # Heatmaps across kappa/tau grid
    for metric in ("r2", "pearson", "lag_samples"):
        fig = plot_grid_heatmaps(grid_metrics, metric=metric)
        if fig is not None:
            all_media[f"grid/{metric}"] = wandb.Image(fig)
            plt.close(fig)

    run.log(all_media)
    for row in all_metrics:
        run.log(row)
    run.finish()
    print(f"\nDone. {len(h5_files)} files × {args.n_samples} samples logged to W&B: {run.url}")


if __name__ == "__main__":
    main()