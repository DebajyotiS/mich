# MICH: Machine Inference for Cortical Haemodynamics

[![CI](https://github.com/DebajyotiS/mich/actions/workflows/ci.yml/badge.svg)](https://github.com/DebajyotiS/mich/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/DebajyotiS/750caf21be803f083ca254dfafc537a2/raw/mich-coverage.json)](https://github.com/DebajyotiS/mich/actions/workflows/ci.yml)

---

## The Problem

Laminar fMRI measures BOLD (Blood Oxygen Level Dependent) signals separately across cortical layers (deep, middle, superficial). These signals are an indirect, blurred, and noisy readout of underlying neural activity -- they are shaped by haemodynamic coupling described by the Balloon model, point-spread contamination across layers, and acquisition noise.

The goal of this project is to invert that process: given observed layer-resolved BOLD signals, recover the latent neural activity that generated them. This is an ill-posed inverse problem. We approach it with a physics-informed neural network (PINN) that jointly fits the data and penalises violations of the Balloon model ODEs, using a collocation-based physics loss over continuous space and time.

---

## Installation

### Prerequisites

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) -- a fast Python package manager that replaces pip + venv

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

### Weights & Biases

Training logs metrics and visualisations to [Weights & Biases](https://wandb.ai/). Create a free account, then log in:
```bash
wandb login
```

### Private config

Local paths and your W&B entity are kept out of version control in `config/private/private.yaml`. Copy the template and fill it in:

```bash
cp config/private/default.yaml config/private/private.yaml
```

Edit `config/private/private.yaml`:
```yaml
entity: <your-wandb-entity>      # W&B username or team name
output_dir: <path/to/outputs>    # where checkpoints and logs are saved
data_dir: <path/to/data>         # where HDF5 simulation files are read from
root_dir: <path/to/repo>         # absolute path to the repo root
```

`private.yaml` is gitignored and never committed.

---

## Usage

### 1. Simulate BOLD data

Generate a synthetic dataset of layer-resolved BOLD signals from simulated neural activity:
```bash
python scripts/run_sim.py config/simulation.yaml
```

This writes an HDF5 file to `data/simulations.h5` (configurable in `config/simulation.yaml`). Each simulation contains neural activity `x`, BOLD signals, and intermediate haemodynamic states.

### 2. Train the inversion model

```bash
python scripts/train_mich.py
```

All settings are controlled via Hydra. Override anything from the command line:
```bash
python scripts/train_mich.py model.heinzle_net.layer_mixing_config.C=32 trainer.max_epochs=100
```

---

## Code Map

### Models -- `src/models/`

| File | What it contains |
|---|---|
| `mich.py` | Top-level `MICH` Lightning module -- training loop, loss, logging |
| `blocks.py` | Network building blocks: `HeinzleNet`, spatial encoder/decoder, temporal mixing, FiLM conditioning |
| `normaliser.py` | `LayerwiseBOLDNormalizer` -- online running-stats normalisation of BOLD inputs |

The main model (`MICH`) takes layer-resolved BOLD as input and predicts the 7 Heinzle haemodynamic states (`x, s, f, v, q, v*, q*`) at each layer, spatial position, and timepoint. It minimises a data loss (predicted BOLD vs. observed) and a physics loss (predicted states vs. Balloon model ODEs) at collocation points sampled across the spatiotemporal domain.

### Data -- `src/data/`

| File | What it contains |
|---|---|
| `balloon.py` | Balloon model ODE RHS, haemodynamic and acquisition constants, BOLD readout |
| `neuronal.py` | Layered neural activity simulator with diffusion and inter-layer drainage |
| `synthetic.py` | `SyntheticBOLDDataset` -- loads HDF5 simulation files for training |
| `signals.py` | Noise models (thermal, physiological) and pulse generation utilities |
| `hcp.py` | HCP data loader (real fMRI) |
| `preprocess.py` | Preprocessing utilities |

### Simulation -- `config/simulation.yaml`

Controls the full forward simulation: grid size, number of layers, TR, haemodynamic constants, pulse parameters, per-layer PSF widths, and noise levels. Running `scripts/run_sim.py` with this config generates the training data.

### Configuration -- `config/`

All configuration is managed by [Hydra](https://hydra.cc/). The entry point is `config/mainconfig.yaml`, which composes:

| Config group | File | Controls |
|---|---|---|
| `model` | `config/model/default.yaml` | Full network architecture |
| `datamodule` | `config/datamodule/default.yaml` | Dataset paths, batch size, splits |
| `trainer` | `config/trainer/default.yaml` | PyTorch Lightning trainer settings |
| `callbacks` | `config/callbacks/default.yaml` | Checkpointing, early stopping |
| `paths` | `config/paths/default.yaml` | Output and data directories |
| `private` | `config/private/private.yaml` | Local paths and W&B entity (gitignored) |

---

## Tests

```bash
pytest -m "not slow and not gpu"
```

Tests live in `tests/`. Slow or GPU-requiring tests are marked and excluded from CI.

---

## License

MIT License
