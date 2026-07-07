"""Harness for the x-error attribution diagnostics (see spec).

Group S (tests/diagnostics/test_structural.py) needs none of this -- it builds its own
tiny SpatioTemporalDecoder fixture. Everything else (Groups P/A/L/O/B) is gated behind
--ckpt: without it, `diag_model` calls pytest.skip(), so `pytest tests/diagnostics` with
no flags only runs Group S.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from common import _forward_and_gather, build_val_loader  # noqa: E402
from eval_mich import load_model, resolve_run  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--ckpt", default=None, help="Path to a trained MICH checkpoint (.ckpt).")
    parser.addoption(
        "--config",
        default=None,
        help="Optional explicit full_config.yaml (default: <ckpt run_dir>/full_config.yaml).",
    )
    parser.addoption("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.addoption(
        "--out",
        default=None,
        help="Where diagnostic JSON/PNG artifacts land (default: <run_dir>/diagnostics/).",
    )


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@pytest.fixture(scope="session")
def diag_device(request) -> torch.device:
    return _resolve_device(request.config.getoption("--device"))


@pytest.fixture(scope="session")
def diag_model(request, diag_device):
    ckpt = request.config.getoption("--ckpt")
    if not ckpt:
        pytest.skip("--ckpt not provided; skipping checkpoint-gated diagnostics")
    run_dir, ckpt_path, full_cfg = resolve_run(ckpt, None, None, "auto")
    cfg_override = request.config.getoption("--config")
    if cfg_override:
        full_cfg = OmegaConf.load(cfg_override)
    model, _model_kind, _ckpt_dict = load_model(full_cfg, ckpt_path, diag_device)
    return model, full_cfg


@pytest.fixture(scope="session")
def diag_out_dir(request, diag_model) -> Path:
    _model, full_cfg = diag_model
    out = request.config.getoption("--out")
    out_dir = Path(out) if out else Path(full_cfg.paths.full_path) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


@pytest.fixture(scope="session")
def diag_val_data(diag_model, diag_device) -> dict[str, torch.Tensor]:
    """One full pass over the val split, gathered at every valid source voxel. Shared
    (session-scoped) across every Group P/A/L/O test so the forward pass runs once."""
    model, full_cfg = diag_model
    loader = build_val_loader(full_cfg)

    collected: dict[str, list[torch.Tensor]] = defaultdict(list)
    for batch in loader:
        gathered = _forward_and_gather(model, batch, diag_device)
        for k, v in gathered.items():
            collected[k].append(v.cpu())

    data: dict[str, torch.Tensor] = {k: torch.cat(v, dim=0) for k, v in collected.items()}
    data["kappa"] = float(model._physio("kappa"))
    data["gamma"] = float(model._physio("gamma"))
    data["burn_in"] = int(getattr(full_cfg.model.loss_config, "burn_in", 0))
    data["max_freq"] = float(full_cfg.model.heinzle_net.time_embedding_config.max_freq)
    data["haemo"] = OmegaConf.to_container(full_cfg.model.haemo, resolve=True)
    data["E0"] = float(full_cfg.model.acquisition.E0)
    data["V0"] = float(full_cfg.model.V0)
    data["order"] = str(full_cfg.model.loss_config.order)
    return data
