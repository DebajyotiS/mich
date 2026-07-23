"""Unit tests for src/mich/utils/hydra_utils.py."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from mich.utils.hydra_utils import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
)
from omegaconf import OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

import wandb

# -----------------------------
# Test doubles / helpers
# -----------------------------


def _write_full_config(path, data, file_name="full_config.yaml"):
    cfg = OmegaConf.create(data)
    OmegaConf.save(cfg, Path(path, file_name))
    return cfg


def _touch(path, mtime=None):
    path.write_text("")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _mk_initialized_mlflow_logger(tmp_path, run_id="mlflow-run-abc"):
    """A real MLFlowLogger with its lazy `.run_id` already resolved, so touching
    it never hits a real store. tracking_uri still points at a real (tmp_path)
    sqlite file because MLFlowLogger.__init__ eagerly creates its MlflowClient,
    which itself eagerly initializes the sqlite store as a side effect."""
    logger = MLFlowLogger(experiment_name="unused", tracking_uri=f"sqlite:///{tmp_path}/unused.db")
    logger._initialized = True
    logger._run_id = run_id
    logger._experiment_id = "exp-1"
    return logger


class _TinyModule(pl.LightningModule):
    """Minimal real LightningModule with a trainable and a frozen layer."""

    def __init__(self):
        super().__init__()
        self.trainable = nn.Linear(3, 4)  # 3*4 + 4 = 16 params
        self.frozen = nn.Linear(4, 2)  # 4*2 + 2 = 10 params
        self.frozen.requires_grad_(False)


# -----------------------------
# reload_original_config
# -----------------------------


def test_reload_original_config_picks_latest_matching_checkpoint(tmp_path):
    _write_full_config(tmp_path, {"foo": 1})
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    old = ckpt_dir / "epoch=01-best.ckpt"
    mid = ckpt_dir / "epoch=02-best.ckpt"
    new = ckpt_dir / "epoch=03-best.ckpt"
    non_matching = ckpt_dir / "epoch=04-last.ckpt"

    # deliberately scramble mtimes so creation order != mtime order
    _touch(old, mtime=1000)
    _touch(new, mtime=3000)
    _touch(mid, mtime=2000)
    _touch(non_matching, mtime=9999)  # newest overall, but doesn't match "*best*"

    cfg = reload_original_config(path=str(tmp_path))
    assert cfg.ckpt_path == str(new)


def test_reload_original_config_raises_indexerror_when_no_checkpoints(tmp_path):
    _write_full_config(tmp_path, {"foo": 1})
    # no checkpoints/ dir at all
    with pytest.raises(IndexError):
        reload_original_config(path=str(tmp_path))


def test_reload_original_config_raises_indexerror_when_no_matching_files(tmp_path):
    _write_full_config(tmp_path, {"foo": 1})
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _touch(ckpt_dir / "epoch=01-last.ckpt")  # doesn't match default "*best*" flag
    with pytest.raises(IndexError):
        reload_original_config(path=str(tmp_path))


def test_reload_original_config_sets_wandb_resume_when_present(tmp_path):
    _write_full_config(tmp_path, {"loggers": {"wandb": {"resume": False}}})
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _touch(ckpt_dir / "best.ckpt")

    cfg = reload_original_config(path=str(tmp_path))
    assert cfg.loggers.wandb.resume is True


def test_reload_original_config_wandb_resume_noop_without_loggers_key(tmp_path):
    _write_full_config(tmp_path, {"foo": 1})
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _touch(ckpt_dir / "best.ckpt")

    cfg = reload_original_config(path=str(tmp_path))  # must not crash
    assert not hasattr(cfg, "loggers")


def test_reload_original_config_wandb_resume_noop_without_wandb_subkey(tmp_path):
    _write_full_config(tmp_path, {"loggers": {"tensorboard": {"save_dir": "x"}}})
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    _touch(ckpt_dir / "best.ckpt")

    cfg = reload_original_config(path=str(tmp_path))  # must not crash
    assert not hasattr(cfg.loggers, "wandb")


def test_reload_original_config_both_flags_false_skip_both_blocks(tmp_path):
    _write_full_config(tmp_path, {"loggers": {"wandb": {"resume": False}}})
    # deliberately no checkpoints/ dir -- if set_ckpt_path ran, this would IndexError
    cfg = reload_original_config(path=str(tmp_path), set_ckpt_path=False, set_wandb_resume=False)
    assert not hasattr(cfg, "ckpt_path")
    assert cfg.loggers.wandb.resume is False


# -----------------------------
# print_config
# -----------------------------


def _full_print_order_cfg():
    return OmegaConf.create(
        {
            "datamodule": {"a": 1},
            "model": {"b": 2},
            "callbacks": {"c": 3},
            "loggers": {"wandb": {"resume": False}},
            "trainer": {"d": 4},
            "paths": {"full_path": "/tmp/x"},
        }
    )


def test_print_config_all_default_fields_present_no_warnings(capsys, caplog):
    cfg = _full_print_order_cfg()
    with caplog.at_level(logging.WARNING):
        print_config(cfg)
    out = capsys.readouterr().out
    for field in ("datamodule", "model", "callbacks", "loggers", "trainer", "paths"):
        assert field in out
    assert not any("not found in config" in r.message for r in caplog.records)


def test_print_config_missing_fields_logs_warning(caplog):
    cfg = OmegaConf.create({"datamodule": {"a": 1}})
    with caplog.at_level(logging.WARNING):
        print_config(cfg)
    warned = [r.message for r in caplog.records if "not found in config" in r.message]
    # model, callbacks, loggers, trainer, paths are all missing from this cfg
    assert any("'model'" in m for m in warned)
    assert any("'paths'" in m for m in warned)
    assert len(warned) == 5


def test_print_config_extra_field_not_in_print_order_is_inserted_and_printed(capsys):
    cfg = OmegaConf.create({"datamodule": {"a": 1}, "custom_extra": {"z": 1}})
    print_config(cfg)
    out = capsys.readouterr().out
    assert "custom_extra" in out
    # extra fields are inserted at the front of the queue (before print_order fields)
    assert out.index("custom_extra") < out.index("datamodule")


def test_print_config_plain_scalar_and_nested_dictconfig_branches(capsys):
    cfg = OmegaConf.create({"paths": "just/a/plain/string", "model": {"nested": {"x": 1}}})
    print_config(cfg)
    out = capsys.readouterr().out
    assert "just/a/plain/string" in out  # str(config_group) branch
    assert "x: 1" in out  # OmegaConf.to_yaml(config_group) branch


# -----------------------------
# save_config
# -----------------------------


def test_save_config_no_loggers_key_just_saves(tmp_path):
    cfg = OmegaConf.create({"paths": {"full_path": str(tmp_path)}, "val": 1})
    save_config(cfg)

    saved_path = tmp_path / "full_config.yaml"
    assert saved_path.exists()
    saved = OmegaConf.load(saved_path)
    assert saved.val == 1
    assert not hasattr(saved, "loggers")


def test_save_config_loggers_without_wandb_subkey_just_saves(tmp_path):
    cfg = OmegaConf.create(
        {"paths": {"full_path": str(tmp_path)}, "loggers": {"tensorboard": {"save_dir": "x"}}}
    )
    save_config(cfg)  # must not crash

    saved = OmegaConf.load(tmp_path / "full_config.yaml")
    assert saved.loggers.tensorboard.save_dir == "x"
    assert not hasattr(saved.loggers, "wandb")


def test_save_config_wandb_run_none_leaves_id_untouched(tmp_path):
    assert wandb.run is None  # default state: nothing in this suite calls wandb.init()
    cfg = OmegaConf.create(
        {"paths": {"full_path": str(tmp_path)}, "loggers": {"wandb": {"id": "orig-id"}}}
    )
    save_config(cfg)

    assert cfg.loggers.wandb.id == "orig-id"
    saved = OmegaConf.load(tmp_path / "full_config.yaml")
    assert saved.loggers.wandb.id == "orig-id"


def test_save_config_sets_wandb_id_from_active_run_and_resolves_interpolations(
    tmp_path, monkeypatch
):
    fake_run = Mock()
    fake_run.id = "run-abc123"
    monkeypatch.setattr(wandb, "run", fake_run)

    cfg = OmegaConf.create(
        {
            "paths": {"full_path": str(tmp_path)},
            "other": {"field": 42},
            "loggers": {"wandb": {"id": "placeholder", "note": "${other.field}"}},
        }
    )
    save_config(cfg)

    # in-memory cfg was mutated with the run id
    assert cfg.loggers.wandb.id == "run-abc123"

    saved_path = tmp_path / "full_config.yaml"
    assert saved_path == tmp_path / "full_config.yaml"
    saved = OmegaConf.load(saved_path)
    assert saved.loggers.wandb.id == "run-abc123"
    # interpolation resolved to its literal value on disk, not left as "${other.field}"
    assert saved.loggers.wandb.note == 42


def test_save_config_mlflow_subkey_untouched_without_logger_passed(tmp_path):
    cfg = OmegaConf.create(
        {"paths": {"full_path": str(tmp_path)}, "loggers": {"mlflow": {"run_id": "orig-id"}}}
    )
    save_config(cfg)  # loggers=None -- no MLFlowLogger instance to read run_id from

    assert cfg.loggers.mlflow.run_id == "orig-id"
    saved = OmegaConf.load(tmp_path / "full_config.yaml")
    assert saved.loggers.mlflow.run_id == "orig-id"


def test_save_config_sets_mlflow_run_id_from_passed_logger(tmp_path):
    mlflow_logger = _mk_initialized_mlflow_logger(tmp_path, run_id="mlflow-run-xyz")
    cfg = OmegaConf.create(
        {"paths": {"full_path": str(tmp_path)}, "loggers": {"mlflow": {"run_id": None}}}
    )
    save_config(cfg, loggers=[mlflow_logger])

    assert cfg.loggers.mlflow.run_id == "mlflow-run-xyz"
    saved = OmegaConf.load(tmp_path / "full_config.yaml")
    assert saved.loggers.mlflow.run_id == "mlflow-run-xyz"


def test_save_config_wandb_and_mlflow_branches_are_independent(tmp_path, monkeypatch):
    fake_run = Mock()
    fake_run.id = "wandb-run-1"
    monkeypatch.setattr(wandb, "run", fake_run)
    mlflow_logger = _mk_initialized_mlflow_logger(tmp_path, run_id="mlflow-run-1")

    cfg = OmegaConf.create(
        {
            "paths": {"full_path": str(tmp_path)},
            "loggers": {"wandb": {"id": "placeholder"}, "mlflow": {"run_id": None}},
        }
    )
    save_config(cfg, loggers=[mlflow_logger])

    assert cfg.loggers.wandb.id == "wandb-run-1"
    assert cfg.loggers.mlflow.run_id == "mlflow-run-1"


# -----------------------------
# log_hyperparameters
# -----------------------------


def test_log_hyperparameters_computes_param_counts_and_logs_once():
    model = _TinyModule()
    trainer = Mock()
    trainer.logger.log_hyperparams = Mock()
    cfg = OmegaConf.create({"a": 1, "b": {"c": 2}})

    log_hyperparameters(cfg, model, trainer)

    trainer.logger.log_hyperparams.assert_called_once()
    hparams = trainer.logger.log_hyperparams.call_args[0][0]

    expected_total = sum(p.numel() for p in model.parameters())
    expected_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected_non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    assert expected_trainable > 0
    assert expected_non_trainable > 0
    assert hparams["model/params/total"] == expected_total
    assert hparams["model/params/trainable"] == expected_trainable
    assert hparams["model/params/non_trainable"] == expected_non_trainable
    # resolved cfg keys are also present in the hparams dict
    assert hparams["a"] == 1
    assert hparams["b"]["c"] == 2


# -----------------------------
# instantiate_collection
# -----------------------------


def test_instantiate_collection_none_returns_empty_list_and_warns(caplog):
    with caplog.at_level(logging.WARNING):
        result = instantiate_collection(None)
    assert result == []
    assert any("empty" in r.message.lower() for r in caplog.records)


def test_instantiate_collection_empty_dictconfig_returns_empty_list(caplog):
    with caplog.at_level(logging.WARNING):
        result = instantiate_collection(OmegaConf.create({}))
    assert result == []
    assert any("empty" in r.message.lower() for r in caplog.records)


def test_instantiate_collection_non_dictconfig_raises_typeerror():
    with pytest.raises(TypeError, match="DictConfig"):
        instantiate_collection({"relu": {"_target_": "torch.nn.ReLU"}})  # plain dict
    with pytest.raises(TypeError, match="DictConfig"):
        instantiate_collection([1, 2, 3])  # plain list


def test_instantiate_collection_instantiates_only_entries_with_target():
    cfg = OmegaConf.create(
        {
            "relu": {"_target_": "torch.nn.ReLU"},
            "no_target": {"foo": "bar"},
            "sigmoid": {"_target_": "torch.nn.Sigmoid"},
        }
    )
    objs = instantiate_collection(cfg)
    assert len(objs) == 2
    assert isinstance(objs[0], torch.nn.ReLU)
    assert isinstance(objs[1], torch.nn.Sigmoid)
