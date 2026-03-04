# MICH: Machine Inference for Cortical Haemodynamics

[![pipeline status](https://gitlab.epfl.ch/sengupta/mich/-/badges/main/pipeline.svg)](https://gitlab.epfl.ch/sengupta/mich/-/commits/main)
[![coverage](https://gitlab.epfl.ch/sengupta/mich/-/badges/main/coverage.svg)](https://gitlab.epfl.ch/sengupta/mich/-/graphs/main/charts)


This project aims to solve the inverse problem: inferring latent neural signals from observed BOLD (Blood Oxygen Level Dependent) signals measured across cortical layers. We use physics-informed neural networks (PINNs) and the Balloon hemodynamic model to address the challenge, focusing on:

- **Identifiability and limits** of BOLD $\rightarrow$ neural inference
- **Robustness and diagnostics** beyond mere reconstruction
- **Reproducible experiments** via Hydra configuration

---

## Getting Started

### 1. Clone the repository
```bash
git clone <repo-url>
cd mich
```

### 2. Prerequisites
- **Python >=3.13**
- **uv** (Python package manager): [uv installation guide](https://github.com/astral-sh/uv)

Install dependencies:
```bash
uv sync
```

### 3. Setup Weights & Biases (wandb)
- Create a [wandb](https://wandb.ai/) account if you don't have one.
- Set your entity/project in `config/private/private.yaml`:

```yaml
entity: <your-wandb-entity>
output_dir: <your-output-dir>
data_dir: <your-data-dir>
root_dir: <your-root-dir>
```

- You may copy and edit `config/private/default.yaml` as a template.

---

## Project Structure

```text
config/           # Hydra configuration hierarchy
  mainconfig.yaml # Top-level config
  private/        # Local/untracked configs (edit private.yaml)
  ...             # Other config folders (callbacks, datamodule, etc.)
data/             # HDF5 simulation datasets
notebooks/        # Jupyter/Markdown notebooks
scripts/          # CLI entry points (run_sim.py, train_mich.py)
src/              # Source code
  data/           # Data generation/loading
  models/         # Model architectures
  utils/          # Utilities
README.md         # This file
pyproject.toml    # Project metadata & dependencies
```

---

## Usage

### Simulate BOLD Data
Run BOLD simulations using a config file:
```bash
python scripts/run_sim.py config/simulation.yaml
```

### Train Inversion Model
Train the PINN model to invert BOLD to neural signals:
```bash
python scripts/train_mich.py
```

- All experiment settings are controlled via Hydra configs in `config/`.
- Outputs and logs are saved to the directory specified in your `private.yaml`.

---

## Configuration System
- All configs are managed via Hydra.
- Main entry: `config/mainconfig.yaml`
- Private/local settings: `config/private/private.yaml`
- Model, data, trainer, and logger configs are modular and composable.

---

## Citation
If you use this codebase, please cite the relevant papers and this repository.

---

## License
MIT License
