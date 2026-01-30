# MICH

**Machine Inference for Cortical Haemodynamics: Physics-informed inverse modeling of BOLD signals and latent neural activity**

This repository contains research code for studying the inverse problem of inferring latent neural activity from BOLD signals, primarily using **physics-informed neural networks (PINNs)** built around the Balloon hemodynamic model.

The emphasis of this project is on:
- Identifiability and limits of BOLD → neural inference
- Robustness and diagnostics, not just reconstruction quality
- Reproducible experiments via Hydra configurations

Core PINN models and the Balloon solver are already implemented. Most work focuses on experimentation, analysis, and extensions.

---

## Project goals

- Invert the Balloon hemodynamic model to recover latent neural drive
- Quantify when the inverse problem is ill-posed or underdetermined
- Compare PINNs against simpler baselines
- Provide diagnostics, benchmarks, and reproducible experiments

A project is considered successful if it **clarifies what is and is not possible**, even if PINNs do not outperform baselines.

---

## Repository structure

```text
.
├── config/                     # Hydra configuration hierarchy
│   ├── callbacks/              # Lightning callbacks
│   ├── data/                   # Dataset configs (synthetic, HCP, etc.)
│   ├── experiments/            # Experiment-level config compositions
│   ├── hydra/                  # Hydra-specific settings
│   ├── loggers/                # Logging backends (wandb, etc.)
│   ├── models/                 # Model and loss configurations
│   ├── paths/                  # Output and data path definitions
│   ├── private/                # Local / untracked configs
│   ├── trainer/                # Lightning trainer configs
│   └── mainconfig.yaml         # Top-level Hydra entry config
│
├── scripts/                    # CLI entry points for training / analysis
│
├── src/
|   |
|   ├── data/               # Data generation and loading
│   │   ├── hcp.py           # HCP BOLD data interface
│   │   └── synthetic.py     # Synthetic Balloon-based data
│   │
│   ├── models/             # Models and losses
│   │   ├── layers.py
│   │   ├── losses.py
│   │   └── pinns.py         # PINN implementations
│   │
│   └── utils/              # Utilities
│       ├── numpy_utils.py
│       ├── plotting.py
│       ├── runtime.py
│       └── torch_utils.py
│
├── tests/                      # Unit and integration tests
├── pyproject.toml
├── README.md
├── uv.lock
└── .python-version
