# MICH: Machine Inference for Cortical Haemodynamics

[![CI](https://github.com/DebajyotiS/mich/actions/workflows/ci.yml/badge.svg)](https://github.com/DebajyotiS/mich/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/DebajyotiS/750caf21be803f083ca254dfafc537a2/raw/mich-coverage.json&cacheSeconds=300)](https://github.com/DebajyotiS/mich/actions/workflows/ci.yml)

---

## The Problem

Laminar fMRI measures BOLD (Blood Oxygen Level Dependent) signals separately across cortical layers (deep, middle, superficial). These signals are an indirect, blurred, and noisy readout of underlying neural activity, shaped by haemodynamic coupling described by the Balloon model, point-spread contamination across layers, and acquisition noise.

The goal of this project is to invert that process: given observed layer-resolved BOLD signals, recover the latent neural activity that generated them. This is an ill-posed inverse problem. We approach it with a physics-informed neural network (PINN) that jointly fits the data and penalises violations of the Balloon model ODEs, using a collocation-based physics loss over continuous space and time.

---

## Installation

### Prerequisites

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv), a fast Python package manager that replaces pip + venv

Install `uv` if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone and install

```bash
git clone git@github.com:DebajyotiS/mich.git
cd mich
```

Create a virtual environment and install the package in editable mode:
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

For development tools (linting, tests):
```bash
uv pip install -e ".[dev]"
```

For notebooks or analysis extras:
```bash
uv pip install -e ".[notebooks,analysis]"
```

> `uv pip install -e .` installs the package from the local source tree in editable mode, so changes to `src/` take effect immediately without reinstalling.

---

## Setup

### Experiment tracking

Training logs metrics, hyperparameters, and visualisations via a Hydra-selectable
logger backend, `loggers=wandb` (default, unchanged) or `loggers=mlflow`:
```bash
python scripts/train_mich.py                  # wandb (default)
python scripts/train_mich.py loggers=mlflow    # mlflow instead
```

**Weights & Biases** (default) logs to [wandb.ai](https://wandb.ai/). Create a free account, then log in:
```bash
wandb login
```

**MLflow** (`loggers=mlflow`) needs no account or login -- it writes to a local
SQLite store at `<output_dir>/<project_name>/mlflow.db`, shared across every
`network_name` run within that project, with artifacts alongside it under
`<output_dir>/<project_name>/mlartifacts/`. Browse runs with:
```bash
mlflow ui --backend-store-uri sqlite:///<output_dir>/<project_name>/mlflow.db
```

### Private config

Local paths (and your W&B entity, if using `loggers=wandb`) are kept out of version control in `config/private/private.yaml`. Copy the template and fill it in:

```bash
cp config/private/default.yaml config/private/private.yaml
```

Edit `config/private/private.yaml`:
```yaml
entity: <your-wandb-entity>      # W&B username or team name (only used by loggers=wandb)
output_dir: <path/to/outputs>    # where checkpoints, logs, and the mlflow store are saved
data_dir: <path/to/data>         # where HDF5 simulation files are read from
root_dir: <path/to/repo>         # absolute path to the repo root
```

`private.yaml` is gitignored and never committed.

---

## Usage

### 1. Simulate BOLD data

`scripts/run_sim.py` is a Hydra entry point (`config_path=config/simulation`, default `config_name=linear`). Each named scenario in `config/simulation/` is a complete, runnable config. There is no single `config/simulation.yaml` to edit:
```bash
python scripts/run_sim.py                                    # default: linear.yaml
python scripts/run_sim.py --config-name=single_layer_single_source
python scripts/run_sim.py --config-name=three_layer_multi_source_multi_active
```

This writes an HDF5 file to the path set by that scenario's `output_path` (e.g. `data/simulations-single_layer_single_source.h5`). See `config/simulation/base.yaml` for the shared defaults every scenario composes from. Each simulation contains neural activity `x`, BOLD signals, and intermediate haemodynamic states, plus a `/meta` group recording the exact simulation config. `train_mich.py` later reads this `/meta` group to keep the model's physics in sync with the data (see below).

### 2. Train the inversion model

```bash
python scripts/train_mich.py
```

All settings are controlled via Hydra. Override anything from the command line:
```bash
python scripts/train_mich.py model.C=32 trainer.max_epochs=100
```

Before instantiating the model, `train_mich.py` reads the `/meta` config stored in the target HDF5 file and overwrites `cfg.model.haemo`, `cfg.model.acquisition`, `cfg.model.L`, `cfg.model.out_channels`, and the decoder's signal list to match. This keeps physiological constants and layer/signal counts in sync with whatever `datamodule.data.path` points to, and is why these values should never be hand-edited in `config/model/*.yaml` when switching datasets.

### 3. Evaluate a trained checkpoint

```bash
python scripts/eval_mich.py --project mich-bold-inversion --network <network_name>
```

`eval_mich.py` is a plain `argparse` script, not a Hydra entry point. It reads `<run_dir>/full_config.yaml` (written by `train_mich.py`) to reconstruct the model and its checkpoint. It can also compare a run against a supervised baseline (`--compare-checkpoint`), or be driven from a saved flat-YAML config in `config/eval/`.

---

## Code Map

### Models (`src/mich/models/`)

| File | What it contains |
|---|---|
| `mich.py` | Top-level `MICH` Lightning module. Composed from the mixins below plus `LightningModule`; owns the training/validation step and the scheduled-loss assembly |
| `mich_losses.py` | `MICHLossMixin`. Every loss term (data, physics, source-activity, quiescence-consistency, supervision and its derivative/phase variants) plus the Gaussian PSF blur. See `training.md` |
| `mich_logging.py` | `MICHLoggingMixin`. Validation-time metrics, plotting, gradient-norm/rank-run hooks (backend-agnostic via `run_adapters.py`) |
| `physio.py` | `LearnablePhysioMixin`. Optionally-learnable physiological constants (kappa/gamma/alpha/tau/V0/E0), parameterised in log-space |
| `collocation.py` | `CollocationMixin`. Collocation-point sampling and gathering into `[B, ..., L, T, H, W]` tensors |
| `blocks.py` | Network building blocks: `HeinzleNet`, spatial encoder/decoder, temporal mixing, FiLM conditioning. See `heinzlenet.md` |
| `normaliser.py` | `LayerwiseBOLDNormalizer`. Online running-stats normalisation of BOLD inputs |
| `supervised.py` | `SupervisedMICH`. A fully-supervised baseline (BOLD to neural activity, no physics constraints), paired with `config/model/supervised.yaml` |

The main model (`MICH`) takes layer-resolved BOLD as input and predicts the Heinzle haemodynamic states (`x, s, f, v, q`, plus `v*, q*` for multi-layer datasets with vascular drainage) at each layer, spatial position, and timepoint. It minimises a data loss (predicted BOLD vs. observed) and a physics loss (predicted states vs. Balloon model ODEs) at collocation points sampled across the spatiotemporal domain. The currently  objective also includes a substantial ground-truth-latent supervision term at the source voxel, alongside a couple of smaller auxiliary losses; all of these are described in `training.md`, including which are on by default and which are opt-in. `SupervisedMICH` is a separate, non-physics baseline used for comparison.

### Data (`src/mich/data/`)

| File | What it contains |
|---|---|
| `balloon.py` | Balloon model ODE RHS, haemodynamic and acquisition constants, BOLD readout |
| `neuronal.py` | Layered neural activity simulator with diffusion and inter-layer drainage |
| `synthetic.py` | `SyntheticH5Dataset` and `SyntheticDataModule` (a `LightningDataModule`). Load HDF5 simulation files for training |
| `signals.py` | Noise models (thermal, physiological) and pulse generation (`ExpDecayPulse`, `RectPulse`, `TriangularPulse`, `SincPulse`, `AlphaPulse`) |

There is currently no real-fMRI (HCP) loader or preprocessing module in the codebase; both are future work, not present files.

### Utilities (`src/mich/utils/`)

| File | What it contains |
|---|---|
| `hydra_utils.py` | Config plumbing used by `train_mich.py`: `instantiate_collection`, `log_hyperparameters`, `save_config`, `reload_original_config` (the resume mechanism, see `lightning_framework.md` §7) |
| `plotting.py` | Validation plot helpers (`plot_neural_bold_layers`, `plot_latent_layers`) used by `mich_logging.py` and `supervised.py` |
| `torch_utils.py` | `get_activation(name)`, mapping strings to `nn.Module` activations |

### Simulation (`config/simulation/`)

Each file is a complete, runnable Hydra config controlling the full forward simulation: grid size, number of layers, TR, haemodynamic constants, pulse parameters, per-layer PSF widths, and noise levels. `base.yaml` holds the shared defaults (3-layer cortex with vascular drainage, 32x32 grid, "exact" Balloon-Windkessel order); the other files (`linear.yaml`, `quadratic.yaml`, `single_layer_single_source.yaml`, `three_layer_multi_source_multi_active.yaml`, etc.) compose from it and override only what differs. Running `scripts/run_sim.py --config-name=<scenario>` generates the training data (see Usage above).

### Configuration (`config/`)

All configuration is managed by [Hydra](https://hydra.cc/). The entry point is `config/mainconfig.yaml`, which composes:

| Config group | File | Controls |
|---|---|---|
| `model` | `config/model/{default,longfreq,supervised}.yaml` | Full network architecture and loss config. `default.yaml` is the 3-layer/full-drainage variant; `longfreq.yaml` is a smaller single-layer variant used by the debug presets below; `supervised.yaml` pairs with `SupervisedMICH` |
| `datamodule` | `config/datamodule/default.yaml` | Dataset paths, batch size, splits |
| `trainer` | `config/trainer/default.yaml` | PyTorch Lightning trainer settings |
| `callbacks` | `config/callbacks/default.yaml` | Checkpointing, LR monitor, model summary |
| `paths` | `config/paths/default.yaml` | Output and data directories |
| `private` | `config/private/private.yaml` | Local paths and W&B entity (gitignored) |
| `loggers` | `config/loggers/{wandb,mlflow}.yaml` | Logger backend settings, selected via `loggers=wandb` (default) or `loggers=mlflow` |
| `hydra` | `config/hydra/default.yaml` | Hydra's own behaviour (working-directory change, run-dir location, log formatting) |
| `experiments` | `config/experiments/{debug_offline,debug_online,default}.yaml` | Optional override bundle applied via `experiments=<name>`, e.g. `experiments=debug_offline` for a fast sanity check |

`config/eval/` (e.g. `single_layer_single_source.yaml`) holds saved flat-YAML configs for `scripts/eval_mich.py`. It is read by that script's own `argparse --config` flag, not composed via Hydra.

See `notebooks/lightning_framework.md` for the full wiring: how these groups compose, what `train_mich.py` actually does step by step, and how to resume a run or launch a Hydra multirun sweep.

---

## Tests

```bash
pytest -m "not slow and not gpu"
```

Unit tests live in `tests/`, one file per source module. `tests/diagnostics/` holds a separate suite of end-to-end diagnostics (lag decomposition, localisation, units correction, ground-truth integration checks, and similar) used to validate training behaviour rather than individual functions. Slow or GPU-requiring tests are marked and excluded from CI.

---

## License

MIT License
