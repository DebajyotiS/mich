"""
Evaluate a trained MICH model on H5 dataset(s).

Usage:
    # Single file:
    python scripts/eval_mich.py checkpoint=<path> data.path=<data.h5>

    # Gridsearch directory:
    python scripts/eval_mich.py checkpoint=<path> data.data_dir=data/gridsearch/
"""

from __future__ import annotations

from pathlib import Path

import h5py
import hydra
import matplotlib.pyplot as plt
import rootutils
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig

root = rootutils.setup_root(__file__, pythonpath=True, cwd=False)

from src.models.mich import MICH
from src.utils.plotting import plot_latent_layers, plot_neural_bold_layers


def load_samples(data_path: str, n: int, device: torch.device) -> dict[str, torch.Tensor]:
    with h5py.File(data_path, "r") as h5f:
        bold = torch.tensor(h5f["layer_0"]["bold"][:n], dtype=torch.float32)
        neural = torch.tensor(h5f["layer_0"]["x"][:n], dtype=torch.float32)
        true_s = torch.tensor(h5f["layer_0"]["s"][:n], dtype=torch.float32)
        true_f = torch.tensor(h5f["layer_0"]["f"][:n], dtype=torch.float32)
        true_v = torch.tensor(h5f["layer_0"]["v"][:n], dtype=torch.float32)
        true_q = torch.tensor(h5f["layer_0"]["q"][:n], dtype=torch.float32)
        src_pos = torch.tensor(h5f["meta"]["source_position"][:n], dtype=torch.long)

    bold = bold.unsqueeze(1).to(device)
    neural = neural.unsqueeze(1).to(device)
    true_s = true_s.unsqueeze(1).to(device)
    true_f = true_f.unsqueeze(1).to(device)
    true_v = true_v.unsqueeze(1).to(device)
    true_q = true_q.unsqueeze(1).to(device)
    src_pos = src_pos.to(device)
    return dict(
        bold=bold,
        neural=neural,
        true_s=true_s,
        true_f=true_f,
        true_v=true_v,
        true_q=true_q,
        src_pos=src_pos,
    )


@torch.no_grad()
def run_inference(model: MICH, bold: torch.Tensor) -> torch.Tensor:
    bold_norm = model.normaliser.normalize(bold) if model.normaliser is not None else bold
    time_grid = model._make_time_grid(
        B=bold.shape[0], T=bold.shape[2], device=bold.device, dtype=bold.dtype
    )
    return model(bold_norm, time_grid, return_gradients=False).z_hat


def eval_one(model: MICH, data_path: str, n_samples: int, device: torch.device, label: str):
    """Run inference on one H5 file, return (bn_images, lat_images) for wandb."""
    data = load_samples(data_path, n_samples, device)
    bold = data["bold"]
    src_pos = data["src_pos"]

    z_hat = run_inference(model, bold)

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

    true_bold = (
        model.normaliser.denormalize(model.normaliser.normalize(bold))
        if model.normaliser is not None
        else bold
    )

    bn_images, lat_images = [], []
    for i in range(n_samples):
        sh, sw = int(src_pos[i, 0]), int(src_pos[i, 1])
        caption = f"{label} | sample {i}"

        fig_bn = plot_neural_bold_layers(
            pred_bold=pred_bold[i, :, :, sh, sw],
            true_bold=true_bold[i, :, :, sh, sw],
            pred_neural=z_hat[i, MICH._signal_index("x"), :, :, sh, sw],
            true_neural=data["neural"][i, :, :, sh, sw],
            source_layer=torch.zeros(1, dtype=torch.long, device=device),
            source_pos=src_pos[i : i + 1],
        )
        bn_images.append(wandb.Image(fig_bn, caption=caption))
        plt.close(fig_bn)

        fig_lat = plot_latent_layers(
            pred_s=z_hat[i, MICH._signal_index("s"), :, :, sh, sw],
            true_s=data["true_s"][i, :, :, sh, sw],
            pred_f=z_hat[i, MICH._signal_index("f"), :, :, sh, sw],
            true_f=data["true_f"][i, :, :, sh, sw],
            pred_v=z_hat[i, MICH._signal_index("v"), :, :, sh, sw],
            true_v=data["true_v"][i, :, :, sh, sw],
            pred_q=z_hat[i, MICH._signal_index("q"), :, :, sh, sw],
            true_q=data["true_q"][i, :, :, sh, sw],
            title=f"{label} — latents",
        )
        lat_images.append(wandb.Image(fig_lat, caption=caption))
        plt.close(fig_lat)

    return bn_images, lat_images


@hydra.main(
    version_base=None,
    config_path=str(root / "config"),
    config_name="evalconfig.yaml",
)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(cfg.checkpoint)

    if cfg.data.path is not None:
        h5_files = [Path(cfg.data.path)]
    else:
        h5_files = sorted(Path(cfg.data.data_dir).glob("*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No H5 files found in {cfg.data.data_dir}")

    print(f"Loading model from {ckpt_path}")
    model = MICH.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model.to(device).eval()

    run_name = cfg.wandb.run_name or f"eval/{ckpt_path.stem}"
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=run_name,
        config={
            "checkpoint": str(ckpt_path),
            "n_samples": cfg.n_samples,
            "n_files": len(h5_files),
        },
    )

    all_bn, all_lat = [], []
    for h5_path in h5_files:
        label = h5_path.stem
        print(f"  evaluating {label} ...")
        bn, lat = eval_one(model, str(h5_path), cfg.n_samples, device, label)
        all_bn.extend(bn)
        all_lat.extend(lat)

    run.log({"media/predictions": all_bn, "media/latents": all_lat})
    run.finish()
    print(f"\nDone. {len(h5_files)} files x {cfg.n_samples} samples logged to W&B: {run.url}")


if __name__ == "__main__":
    main()
