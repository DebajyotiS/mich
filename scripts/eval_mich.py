"""
Evaluate a trained MICH (PINN) or SupervisedMICH checkpoint.

Model resolution is done by either:
    --project <name> --network <name>   resolves <output_dir>/<project>/<network>/
    -c/--checkpoint <path.ckpt>         resolves run_dir = checkpoint.parent.parent

Either way, `<run_dir>/full_config.yaml` (written by scripts/train_mich.py) is required
it's the only place that records what the checkpoint was actually trained with (data path,
architecture, physics constants). Checkpoints without it aren't supported.

Data resolution — either:
    -d/--data <file_or_dir> [<file_or_dir> ...]   one or more H5 files / directories of H5 files
    (nothing)                                     falls back to the run's own held-out test split

Examples:
    # by name, evaluated on its own test split
    python scripts/eval_mich.py --project mich-bold-inversion --network single_layer_single_source

    # direct checkpoint, evaluated on specific files
    python scripts/eval_mich.py -c results/.../checkpoints/best.ckpt -d data/gridsearch/

    # compare a PINN and a supervised baseline on the same data
    python scripts/eval_mich.py --project P --network pinn_run \
        --compare-checkpoint results/P/supervised_run/checkpoints/best.ckpt -d data/a.h5

    # from a saved config (see config/eval/*.yaml) -- CLI flags override individual keys
    python scripts/eval_mich.py --config config/eval/single_layer_single_source.yaml
    python scripts/eval_mich.py --config config/eval/single_layer_single_source.yaml -n 16
"""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rich
import rich.console
import rich.markup
import torch
from hydra.utils import instantiate
from matplotlib.animation import FuncAnimation, PillowWriter
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import default_collate

import wandb
from mich import CONFIG_DIR
from mich.data.synthetic import SyntheticH5Dataset, discover_layers
from mich.utils.plotting import LAYER_NAMES, plot_latent_layers, plot_neural_bold_layers

console = rich.console.Console()


def _pick_checkpoint(ckpt_dir: Path, tag: str) -> Path:
    def _latest(pattern: str) -> Path | None:
        matches = sorted(ckpt_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        return matches[-1] if matches else None

    candidates = {
        "best": ["best*.ckpt"],
        "last": ["last*.ckpt"],
        "auto": ["best*.ckpt", "last*.ckpt"],
    }
    for pattern in candidates[tag]:
        ckpt = _latest(pattern)
        if ckpt is not None:
            return ckpt
    raise FileNotFoundError(f"No checkpoint matching {candidates[tag]} found in {ckpt_dir}")


def resolve_run(checkpoint: str | None, project: str | None, network: str | None, ckpt_tag: str):
    """Returns (run_dir, ckpt_path, full_cfg)."""
    if checkpoint:
        ckpt_path = Path(checkpoint).resolve()
        run_dir = ckpt_path.parent.parent
    else:
        if not (project and network):
            raise SystemExit("Provide -c/--checkpoint, or both --project and --network.")
        private_cfg = OmegaConf.load(CONFIG_DIR / "private" / "private.yaml")
        run_dir = Path(private_cfg.output_dir) / project / network
        ckpt_path = _pick_checkpoint(run_dir / "checkpoints", ckpt_tag)

    full_cfg_path = run_dir / "full_config.yaml"
    if not full_cfg_path.exists():
        raise FileNotFoundError(
            f"No full_config.yaml at {full_cfg_path}. This script reads that file for the "
            "model architecture and training provenance; checkpoints not produced by "
            "scripts/train_mich.py (or missing that file) aren't supported."
        )
    full_cfg = OmegaConf.load(full_cfg_path)
    return run_dir, ckpt_path, full_cfg


def load_model(full_cfg: DictConfig, ckpt_path: Path, device: torch.device):
    target = full_cfg.model._target_
    model_kind = "supervised" if "supervised" in target.lower() else "pinn"
    model = instantiate(full_cfg.model)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, model_kind, ckpt


def describe_run(
    run_dir: Path, ckpt_path: Path, full_cfg: DictConfig, model_kind: str, ckpt: dict
) -> dict:
    dm_data = full_cfg.datamodule.data
    layers = dm_data.get("layers")
    info = {
        "project": run_dir.parent.name,
        "network": run_dir.name,
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
        "model_kind": model_kind,
        "model_target": full_cfg.model._target_,
        "trained_on_data_path": str(dm_data.get("path")),
        "trained_on_layers": OmegaConf.to_container(layers)
        if OmegaConf.is_config(layers)
        else layers,
        "L": full_cfg.model.get("L"),
        "out_channels": full_cfg.model.get("out_channels"),
    }
    haemo = full_cfg.model.get("haemo")
    if haemo:
        info["haemo"] = OmegaConf.to_container(haemo, resolve=True)
    acquisition = full_cfg.model.get("acquisition")
    if acquisition:
        info["acquisition"] = OmegaConf.to_container(acquisition, resolve=True)
    return info


def print_banner(info: dict) -> None:
    console.rule("[bold]Model provenance — what this checkpoint was trained on")
    for k, v in info.items():
        console.print(f"[cyan]{k}[/]: {rich.markup.escape(str(v))}")
    console.rule()


def _expand_data_args(data_args: list[str]) -> list[Path]:
    files: list[Path] = []
    for entry in data_args:
        p = Path(entry)
        if p.is_dir():
            files.extend(sorted(p.glob("*.h5")))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(f"Data path not found: {entry}")
    if not files:
        raise FileNotFoundError(f"No .h5 files found for: {data_args}")
    return files


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def load_file_batch(h5_path: Path, n: int, h5_cache_cfg: dict, device: torch.device) -> dict:
    """Loads the first `n` samples of an H5 file via the same Dataset class used for training,
    so multi-layer files and metadata are handled identically to the training pipeline."""
    layers = discover_layers(str(h5_path))
    try:
        ds = SyntheticH5Dataset(
            path=str(h5_path),
            cache_cfg=h5_cache_cfg,
            layers=layers,
            dtype=torch.float32,
            return_meta=True,
            return_latents=True,
        )
    except KeyError:
        # file has no latent (s/f/v/q) groups -- fall back to bold/neural only
        ds = SyntheticH5Dataset(
            path=str(h5_path),
            cache_cfg=h5_cache_cfg,
            layers=layers,
            dtype=torch.float32,
            return_meta=True,
            return_latents=False,
        )
    n = min(n, len(ds))
    batch = default_collate([ds[i] for i in range(n)])
    return _to_device(batch, device)


def load_test_split_batch(full_cfg: DictConfig, n: int, device: torch.device) -> tuple[dict, str]:
    """Falls back to the model's own held-out test split (from its training config)."""
    dm_cfg = deepcopy(full_cfg.datamodule)
    with open_dict(dm_cfg):
        dm_cfg.data.return_latents = True
        dm_cfg.data.return_meta = True
        dm_cfg.data.dtype = "float32"  # model params are fp32; training may use fp16 storage
    datamodule = instantiate(dm_cfg)
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    collected, got = [], 0
    for batch in loader:
        collected.append(batch)
        got += batch["bold"].shape[0]
        if got >= n:
            break
    if not collected:
        raise RuntimeError("Test split is empty -- check split.test_frac in the training config.")

    merged = {k: torch.cat([b[k] for b in collected], dim=0)[:n] for k in collected[0]}
    label = f"{Path(str(full_cfg.datamodule.data.path)).stem}__test_split"
    return _to_device(merged, device), label


@torch.no_grad()
def infer_pinn(model, batch: dict):
    """Returns (z_hat [B,7,L,T,H,W], pred_neural [B,L,T,H,W], pred_bold [B,L,T,H,W])."""
    bold = batch["bold"]
    source_position = batch["source_position"]
    num_sources = batch["num_sources"]

    bold_norm = (
        model.normaliser(bold, source_position, num_sources)
        if model.normaliser is not None
        else bold
    )
    time_grid = model._make_time_grid(
        B=bold.shape[0], T=bold.shape[2], device=bold.device, dtype=bold.dtype
    )
    manifest = model(bold_norm, time_grid, return_gradients=False, normalise=False)
    z_hat = manifest.z_hat

    pred_bold = model._compute_bold(
        z_hat[:, model._signal_index("v")],
        z_hat[:, model._signal_index("q")],
        acquisition=model._current_acquisition(),
        V0=model._physio("V0"),
    )
    pred_bold = model._apply_psf_blur(pred_bold)
    pred_neural = z_hat[:, model._signal_index("x")]
    return z_hat, pred_neural, pred_bold


@torch.no_grad()
def infer_supervised(model, batch: dict):
    """Returns pred_neural [B,L,T,H,W]."""
    return model(batch["bold"])


def gather_source_traces(
    full: torch.Tensor, source_position: torch.Tensor, num_sources: torch.Tensor
):
    """full: [B, L, T, H, W] -> [sum(valid sources), L, T], gathered at every real source."""
    B, S = source_position.shape[:2]
    src_h = source_position[..., 0].long()
    src_w = source_position[..., 1].long()
    b_idx = torch.arange(B, device=full.device).unsqueeze(1).expand(-1, S)
    gathered = full[b_idx, :, :, src_h, src_w]  # [B, S, L, T]
    mask = torch.arange(S, device=full.device)[None, :] < num_sources[:, None]
    return gathered[mask]


def grid_metrics_fn(model_cls) -> callable:
    return model_cls._neural_recovery_metrics


def compute_grid_metrics(
    pred_full: torch.Tensor, true_full: torch.Tensor, metrics_fn, max_rows: int = 20000
):
    """Same R2/Pearson/lag metric, but over every spatial voxel (not just source voxels) --
    exposes whether the model hallucinates activity in silent background regions."""
    B, L, T, H, W = true_full.shape
    pred = pred_full.permute(0, 1, 3, 4, 2).reshape(-1, T)
    true = true_full.permute(0, 1, 3, 4, 2).reshape(-1, T)
    if pred.shape[0] > max_rows:
        idx = torch.randperm(pred.shape[0], device=pred.device)[:max_rows]
        pred, true = pred[idx], true[idx]
    return metrics_fn(pred, true)


def make_evolution_gif(
    true_grid: np.ndarray,  # [L, T, H, W]
    pred_grid: np.ndarray,  # [L, T, H, W]
    out_path: Path,
    *,
    signal_name: str,
    cmap: str = "magma",
    fps: int = 6,
) -> Path:
    L, T, H, W = true_grid.shape
    err_grid = true_grid - pred_grid
    vmin, vmax = np.quantile(true_grid, [0.01, 0.99])
    err_abs = float(np.quantile(np.abs(err_grid), 0.99)) or 1e-6

    fig, axes = plt.subplots(L, 3, figsize=(9, 3 * L), constrained_layout=True, squeeze=False)
    col_titles = ["True", "Predicted", "Error (true - pred)"]
    panels = [
        (true_grid, cmap, vmin, vmax),
        (pred_grid, cmap, vmin, vmax),
        (err_grid, "RdBu_r", -err_abs, err_abs),
    ]
    ims = [[None] * 3 for _ in range(L)]

    for row in range(L):
        layer_idx = L - 1 - row  # match plot_neural_bold_layers: row 0 = most superficial
        for col, (arr, cm, vlo, vhi) in enumerate(panels):
            im = axes[row][col].imshow(
                arr[layer_idx, 0], cmap=cm, vmin=vlo, vmax=vhi, aspect="auto"
            )
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
            if row == 0:
                axes[row][col].set_title(col_titles[col], fontfamily="monospace")
            if col == 0:
                name = LAYER_NAMES[row] if row < len(LAYER_NAMES) else f"layer {layer_idx}"
                axes[row][col].set_ylabel(name, fontfamily="monospace")
            ims[row][col] = im
    fig.suptitle(f"{signal_name} evolution", fontfamily="monospace")

    def update(t):
        artists = []
        for row in range(L):
            layer_idx = L - 1 - row
            ims[row][0].set_data(true_grid[layer_idx, t])
            ims[row][1].set_data(pred_grid[layer_idx, t])
            ims[row][2].set_data(err_grid[layer_idx, t])
            artists.extend(ims[row])
        return artists

    ani = FuncAnimation(fig, update, frames=T, blit=False)
    ani.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def plot_prediction_figures(batch, pred_neural, pred_bold, n_samples: int):
    source_position = batch["source_position"]
    figs = []
    for i in range(min(n_samples, batch["bold"].shape[0])):
        h, w = int(source_position[i, 0, 0]), int(source_position[i, 0, 1])
        fig = plot_neural_bold_layers(
            pred_bold=pred_bold[i, :, :, h, w].float(),
            true_bold=batch["bold"][i, :, :, h, w].float(),
            pred_neural=pred_neural[i, :, :, h, w].float(),
            true_neural=batch["neural"][i, :, :, h, w].float(),
            source_layer=batch["source_layer"][i],
            source_pos=batch["source_position"][i],
            num_sources=batch["num_sources"][i],
        )
        figs.append(fig)
    return figs


def plot_latent_figures(model, z_hat, batch, n_samples: int):
    if "s" not in batch:
        return []
    source_position = batch["source_position"]
    has_drain = z_hat.shape[1] > 5
    figs = []
    for i in range(min(n_samples, z_hat.shape[0])):
        h, w = int(source_position[i, 0, 0]), int(source_position[i, 0, 1])
        kwargs = dict(
            pred_s=z_hat[i, model._signal_index("s"), :, :, h, w].float(),
            true_s=batch["s"][i, :, :, h, w].float(),
            pred_f=z_hat[i, model._signal_index("f"), :, :, h, w].float(),
            true_f=batch["f"][i, :, :, h, w].float(),
            pred_v=z_hat[i, model._signal_index("v"), :, :, h, w].float(),
            true_v=batch["v"][i, :, :, h, w].float(),
            pred_q=z_hat[i, model._signal_index("q"), :, :, h, w].float(),
            true_q=batch["q"][i, :, :, h, w].float(),
            title="Latent States",
        )
        if has_drain and "v_star" in batch:
            kwargs.update(
                pred_v_star=z_hat[i, model._signal_index("vstar"), :, :, h, w].float(),
                true_v_star=batch["v_star"][i, :, :, h, w].float(),
                pred_q_star=z_hat[i, model._signal_index("qstar"), :, :, h, w].float(),
                true_q_star=batch["q_star"][i, :, :, h, w].float(),
            )
        figs.append(plot_latent_layers(**kwargs))
    return figs


def _parse_kappa_tau(stem: str):
    m = re.search(r"kappa([0-9.]+)_tau([0-9.]+)", stem)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def plot_grid_heatmaps(grid_metrics: dict[str, dict[str, float]], metric: str = "r2"):
    entries = []
    for stem, vals in grid_metrics.items():
        kappa, tau = _parse_kappa_tau(stem)
        if kappa is not None:
            entries.append((kappa, tau, vals))
    if not entries:
        return None

    kappas = sorted(set(e[0] for e in entries))
    taus = sorted(set(e[1] for e in entries))
    k_idx = {k: i for i, k in enumerate(kappas)}
    t_idx = {t: i for i, t in enumerate(taus)}
    models = [p for p in ("pinn", "supervised") if any(f"{p}/{metric}" in v for _, _, v in entries)]
    if not models:
        return None

    grids, all_vals = {}, []
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
    for ax, prefix in zip(axes, models, strict=False):
        grid = grids[prefix]
        im = ax.imshow(grid, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="RdYlGn")
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
                    ax.text(
                        ti,
                        ki,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if 0.2 < v < 0.85 else "white",
                    )
    fig.suptitle(f"Neural recovery {metric} across haemodynamic grid", fontsize=12)
    return fig


def evaluate_one(model, model_kind: str, batch: dict, label: str, args, out_dir: Path | None):
    metrics_fn = grid_metrics_fn(type(model))
    is_pinn = model_kind == "pinn"

    if is_pinn:
        z_hat, pred_neural, pred_bold = infer_pinn(model, batch)
    else:
        pred_neural = infer_supervised(model, batch)
        pred_bold, z_hat = None, None

    source_metrics = metrics_fn(
        gather_source_traces(pred_neural, batch["source_position"], batch["num_sources"]),
        gather_source_traces(batch["neural"], batch["source_position"], batch["num_sources"]),
    )
    grid_metrics = compute_grid_metrics(pred_neural, batch["neural"], metrics_fn)
    metrics = {
        f"{model_kind}/source/r2": source_metrics["val/neural/r2"],
        f"{model_kind}/source/pearson": source_metrics["val/neural/pearson"],
        f"{model_kind}/source/lag_samples": source_metrics["val/neural/lag_samples"],
        f"{model_kind}/grid/r2": grid_metrics["val/neural/r2"],
        f"{model_kind}/grid/pearson": grid_metrics["val/neural/pearson"],
    }

    pred_figs = plot_prediction_figures(
        batch, pred_neural, pred_bold if pred_bold is not None else batch["bold"], args.n_samples
    )
    latent_figs = plot_latent_figures(model, z_hat, batch, args.n_samples) if is_pinn else []

    gif_paths = []
    if not args.no_gif:
        for i in range(min(args.n_gif_samples, pred_neural.shape[0])):
            true_neural_np = batch["neural"][i].cpu().numpy()
            pred_neural_np = pred_neural[i].cpu().numpy()
            gif_dir = out_dir if out_dir is not None else Path(".eval_tmp")
            gif_dir.mkdir(parents=True, exist_ok=True)
            neural_gif = gif_dir / f"{label}_{model_kind}_sample{i}_neural.gif"
            make_evolution_gif(
                true_neural_np, pred_neural_np, neural_gif, signal_name="Neural", cmap="magma"
            )
            gif_paths.append(neural_gif)
            if is_pinn:
                true_bold_np = batch["bold"][i].cpu().numpy()
                pred_bold_np = pred_bold[i].cpu().numpy()
                bold_gif = gif_dir / f"{label}_{model_kind}_sample{i}_bold.gif"
                make_evolution_gif(
                    true_bold_np, pred_bold_np, bold_gif, signal_name="BOLD", cmap="coolwarm"
                )
                gif_paths.append(bold_gif)

    if out_dir is not None:
        for i, fig in enumerate(pred_figs):
            fig.savefig(out_dir / f"{label}_{model_kind}_predictions_{i}.png", dpi=120)
        for i, fig in enumerate(latent_figs):
            fig.savefig(out_dir / f"{label}_{model_kind}_latents_{i}.png", dpi=120)
        with open(out_dir / f"{label}_{model_kind}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics, pred_figs, latent_figs, gif_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MICH / SupervisedMICH checkpoints.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML file of default args (see config/eval/*.yaml). Explicit CLI flags override it.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Project name (with --network, resolves the checkpoint path)",
    )
    parser.add_argument(
        "--network",
        default=None,
        help="Network name (with --project, resolves the checkpoint path)",
    )
    parser.add_argument("--ckpt-tag", default="auto", choices=["best", "last", "auto"])
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Direct checkpoint path (overrides --project/--network)",
    )
    parser.add_argument(
        "--compare-checkpoint",
        default=None,
        help="Optional second checkpoint evaluated on the same data",
    )
    parser.add_argument(
        "-d",
        "--data",
        nargs="+",
        default=None,
        help="H5 file(s) or director(ies); default: the run's own test split",
    )
    parser.add_argument(
        "-n",
        "--n-samples",
        type=int,
        default=8,
        help="Samples loaded per data source, used for metrics + plots",
    )
    parser.add_argument(
        "--n-gif-samples",
        type=int,
        default=1,
        help="How many of those samples also get a gif (expensive)",
    )
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--local-out",
        default=None,
        help="Local output dir (default: <run_dir>/results/ next to the checkpoint being evaluated)",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    return parser


def parse_args() -> argparse.Namespace:
    """Two-pass parse: if --config is given, its keys become the parser's defaults,
    so any flag explicitly passed on the CLI still overrides the file."""
    parser = build_parser()
    prelim, _ = parser.parse_known_args()
    if prelim.config:
        cfg = OmegaConf.to_container(OmegaConf.load(prelim.config), resolve=True)
        known = {a.dest for a in parser._actions}
        unknown = set(cfg) - known
        if unknown:
            raise SystemExit(f"Unknown key(s) in {prelim.config}: {sorted(unknown)}")
        parser.set_defaults(**cfg)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir, ckpt_path, full_cfg = resolve_run(
        args.checkpoint, args.project, args.network, args.ckpt_tag
    )
    console.print(f"Loading {ckpt_path}")
    model, model_kind, ckpt = load_model(full_cfg, ckpt_path, device)
    banner = describe_run(run_dir, ckpt_path, full_cfg, model_kind, ckpt)
    print_banner(banner)

    compare_model, compare_kind, compare_banner = None, None, None
    if args.compare_checkpoint:
        c_run_dir, c_ckpt_path, c_full_cfg = resolve_run(
            args.compare_checkpoint, None, None, args.ckpt_tag
        )
        console.print(f"Loading comparison checkpoint {c_ckpt_path}")
        compare_model, compare_kind, c_ckpt = load_model(c_full_cfg, c_ckpt_path, device)
        compare_banner = describe_run(c_run_dir, c_ckpt_path, c_full_cfg, compare_kind, c_ckpt)
        print_banner(compare_banner)

    # results/ lives next to the checkpoint being evaluated: per-datafile output goes in its
    # own subfolder; anything that compares across datafiles (e.g. the kappa/tau heatmap)
    # goes in the results/ root since it isn't specific to any single one.
    local_root = Path(args.local_out) if args.local_out else run_dir / "results"
    local_root.mkdir(parents=True, exist_ok=True)
    with open(local_root / "provenance.json", "w") as f:
        json.dump({"primary": banner, "compare": compare_banner}, f, indent=2)
    console.print(f"Local outputs: {local_root}")

    run = None
    if not args.no_wandb:
        run = wandb.init(
            project=args.wandb_project or run_dir.parent.name,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"eval/{run_dir.name}",
            config={"primary": banner, "compare": compare_banner},
        )

    h5_cache_cfg = OmegaConf.to_container(full_cfg.datamodule.h5_cache, resolve=True)

    if args.data:
        h5_files = _expand_data_args(args.data)
        data_sources = [
            (p.stem, lambda p=p: load_file_batch(p, args.n_samples, h5_cache_cfg, device))
            for p in h5_files
        ]
    else:
        console.print("No -d/--data given -- falling back to the run's own held-out test split.")
        data_sources = [
            ("test_split", lambda: load_test_split_batch(full_cfg, args.n_samples, device)[0])
        ]

    all_media, grid_metrics = {}, {}
    for label, load_fn in data_sources:
        console.print(f"  evaluating {label} ...")
        batch = load_fn()
        out_dir = local_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        grid_metrics.setdefault(label, {})

        for kind, m in (
            (model_kind, model),
            *([(compare_kind, compare_model)] if compare_model else []),
        ):
            metrics, pred_figs, latent_figs, gif_paths = evaluate_one(
                m, kind, batch, label, args, out_dir
            )
            grid_metrics[label].update(metrics)
            summary = "  ".join(f"{k.split('/', 1)[1]}={v:.3f}" for k, v in metrics.items())
            console.print(f"    {kind}: {summary}")

            if run is not None:
                all_media.setdefault(f"{kind}/{label}/predictions", []).extend(
                    wandb.Image(fig) for fig in pred_figs
                )
                if latent_figs:
                    all_media.setdefault(f"{kind}/{label}/latents", []).extend(
                        wandb.Image(fig) for fig in latent_figs
                    )
                for gif_path in gif_paths:
                    all_media[f"{kind}/{label}/gif/{gif_path.stem}"] = wandb.Video(
                        str(gif_path), fps=6, format="gif"
                    )
                run.log({f"{kind}/{label}/{k.split('/', 1)[1]}": v for k, v in metrics.items()})

            for fig in pred_figs + latent_figs:
                plt.close(fig)

    # Cross-datafile comparison output goes in the results/ root (not any one file's subfolder):
    # a plain metrics table always, plus a kappa/tau heatmap when filenames follow that convention.
    with open(local_root / "metrics_summary.json", "w") as f:
        json.dump(grid_metrics, f, indent=2)

    for metric in ("r2", "pearson"):
        fig = plot_grid_heatmaps(grid_metrics, metric=metric)
        if fig is not None:
            fig.savefig(local_root / f"grid_{metric}.png", dpi=120)
            if run is not None:
                all_media[f"grid/{metric}"] = wandb.Image(fig)
            plt.close(fig)

    if run is not None:
        run.log(all_media)
        run.finish()
        console.print(f"W&B run: {run.url}")

    console.print(
        f"\nDone. {len(data_sources)} data source(s) evaluated. Local outputs: {local_root}"
    )


if __name__ == "__main__":
    main()
