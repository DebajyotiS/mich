"""Unit tests for MICHLoggingMixin (src/mich/models/mich_logging.py)."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest
import torch
from mich.models import mich_logging
from mich.models.mich_logging import MICHLoggingMixin
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger


class _LoggingHost(MICHLoggingMixin):
    """Minimal concrete object mixing in MICHLoggingMixin, enough to call every
    hook directly without a real Trainer/HeinzleNet."""

    def __init__(self, *, trainer=None, global_step=0, global_rank=0, heinzle_net=None):
        self.trainer = trainer
        self.global_step = global_step
        self.global_rank = global_rank
        self.heinzle_net = heinzle_net
        self.logged: list[tuple[str, object, dict]] = []

    def log(self, name, value, **kwargs):
        self.logged.append((name, value, kwargs))


def _mk_trainer(*, is_global_zero=True, log_every_n_steps=5, logger=None):
    return types.SimpleNamespace(
        is_global_zero=is_global_zero, log_every_n_steps=log_every_n_steps, logger=logger
    )


def _mk_heinzle_net_double(n_heads=3):
    """Fake heinzle_net.spatial_decoder with real nn.Modules so .parameters()/.grad work."""
    time_film = types.SimpleNamespace(linear=torch.nn.Linear(2, 2), out=torch.nn.Linear(2, 2))
    out_heads = [torch.nn.Conv2d(1, 1, 1) for _ in range(n_heads)]
    spatial_decoder = types.SimpleNamespace(time_film=time_film, out_heads=out_heads)
    return types.SimpleNamespace(spatial_decoder=spatial_decoder)


def _set_grad(module, value=1.0):
    for p in module.parameters():
        p.grad = torch.full_like(p, value)


# -----------------------------
# _neural_recovery_metrics
# -----------------------------


def test_neural_recovery_metrics_identical_signals_perfect_scores():
    x = torch.randn(2, 3, 20)
    m = MICHLoggingMixin._neural_recovery_metrics(x, x.clone())
    assert m["val/neural/r2"] == pytest.approx(1.0, abs=1e-4)
    assert m["val/neural/pearson"] == pytest.approx(1.0, abs=1e-4)
    assert m["val/neural/lag_samples"] == pytest.approx(0.0, abs=1e-4)


def test_neural_recovery_metrics_detects_constant_lag_shift():
    T = 40
    true = torch.zeros(1, 1, T)
    true[0, 0, T // 2] = 1.0  # single spike
    shift = 5
    pred = torch.zeros(1, 1, T)
    pred[0, 0, T // 2 - shift] = 1.0  # pred leads true by `shift` samples

    m = MICHLoggingMixin._neural_recovery_metrics(pred, true)
    assert abs(m["val/neural/lag_samples"]) == pytest.approx(shift, abs=1e-4)


def test_neural_recovery_metrics_returns_expected_keys_and_finite():
    pred = torch.randn(3, 2, 15)
    true = torch.randn(3, 2, 15)
    m = MICHLoggingMixin._neural_recovery_metrics(pred, true)
    assert set(m.keys()) == {"val/neural/r2", "val/neural/pearson", "val/neural/lag_samples"}
    assert all(torch.isfinite(torch.tensor(v)) for v in m.values())


# -----------------------------
# _plot_and_log_predictions / _plot_and_log_latents / _plot_and_log_x_recon
# -----------------------------


def test_plot_and_log_predictions_noop_when_no_adapter():
    host = _LoggingHost(global_step=10)  # no _adapter attribute set
    B, L, T = 2, 2, 5
    host._plot_and_log_predictions(
        pred_bold=torch.randn(B, L, T),
        true_bold=torch.randn(B, L, T),
        pred_neural=torch.randn(B, L, T),
        true_neural=torch.randn(B, L, T),
        source_layer=torch.zeros(B, 1, dtype=torch.long),
        source_pos=torch.zeros(B, 1, 2, dtype=torch.long),
        num_sources=torch.ones(B, dtype=torch.long),
    )  # must not raise


def test_plot_and_log_predictions_logs_one_image_per_sample():
    host = _LoggingHost(global_step=10)
    host._adapter = MagicMock()
    B, L, T = 3, 2, 5
    host._plot_and_log_predictions(
        pred_bold=torch.randn(B, L, T),
        true_bold=torch.randn(B, L, T),
        pred_neural=torch.randn(B, L, T),
        true_neural=torch.randn(B, L, T),
        source_layer=torch.zeros(B, 1, dtype=torch.long),
        source_pos=torch.zeros(B, 1, 2, dtype=torch.long),
        num_sources=torch.ones(B, dtype=torch.long),
    )
    host._adapter.log.assert_called_once()
    (payload,), kwargs = host._adapter.log.call_args
    assert payload["global_step"] == 10
    assert len(payload["media/predictions"]) == B
    assert kwargs.get("commit") is False
    # No voxel_pos/is_source_voxel given -- no suptitle set, matches pre-existing behaviour.
    assert payload["media/predictions"][0]._suptitle is None


def test_plot_and_log_predictions_sets_suptitle_from_voxel_kind():
    host = _LoggingHost(global_step=10)
    host._adapter = MagicMock()
    B, L, T = 2, 2, 5
    host._plot_and_log_predictions(
        pred_bold=torch.randn(B, L, T),
        true_bold=torch.randn(B, L, T),
        pred_neural=torch.randn(B, L, T),
        true_neural=torch.randn(B, L, T),
        source_layer=torch.zeros(B, 1, dtype=torch.long),
        source_pos=torch.zeros(B, 1, 2, dtype=torch.long),
        num_sources=torch.ones(B, dtype=torch.long),
        voxel_pos=torch.tensor([[1, 2], [3, 4]]),
        is_source_voxel=torch.tensor([True, False]),
    )
    (payload,), _ = host._adapter.log.call_args
    images = payload["media/predictions"]
    assert images[0]._suptitle.get_text() == "Source voxel @ (1, 2)"
    assert images[1]._suptitle.get_text() == "Off-source voxel @ (3, 4)"


def test_plot_and_log_latents_without_drain_logs_and_commits():
    host = _LoggingHost(global_step=20)
    host._adapter = MagicMock()
    B, L, T = 2, 2, 5
    kwargs = {
        k: torch.randn(B, L, T)
        for k in ("pred_s", "true_s", "pred_f", "true_f", "pred_v", "true_v", "pred_q", "true_q")
    }
    host._plot_and_log_latents(**kwargs)
    host._adapter.log.assert_called_once()
    (payload,), call_kwargs = host._adapter.log.call_args
    assert len(payload["media/latents"]) == B
    assert call_kwargs.get("commit") is True
    # No voxel_pos/is_source_voxel given -- title falls back to the plain default.
    assert payload["media/latents"][0]._suptitle.get_text() == "Latent States"


def test_plot_and_log_latents_sets_title_from_voxel_kind():
    host = _LoggingHost(global_step=20)
    host._adapter = MagicMock()
    B, L, T = 2, 2, 5
    kwargs = {
        k: torch.randn(B, L, T)
        for k in ("pred_s", "true_s", "pred_f", "true_f", "pred_v", "true_v", "pred_q", "true_q")
    }
    host._plot_and_log_latents(
        **kwargs,
        voxel_pos=torch.tensor([[1, 2], [3, 4]]),
        is_source_voxel=torch.tensor([True, False]),
    )
    (payload,), _ = host._adapter.log.call_args
    images = payload["media/latents"]
    assert images[0]._suptitle.get_text() == "Latent States -- Source voxel @ (1, 2)"
    assert images[1]._suptitle.get_text() == "Latent States -- Off-source voxel @ (3, 4)"


def test_plot_and_log_latents_with_drain_branch_runs():
    host = _LoggingHost(global_step=20)
    host._adapter = MagicMock()
    B, L, T = 2, 2, 5
    names = (
        "pred_s",
        "true_s",
        "pred_f",
        "true_f",
        "pred_v",
        "true_v",
        "pred_q",
        "true_q",
        "pred_v_star",
        "true_v_star",
        "pred_q_star",
        "true_q_star",
    )
    kwargs = {k: torch.randn(B, L, T) for k in names}
    host._plot_and_log_latents(**kwargs)
    host._adapter.log.assert_called_once()


def test_plot_and_log_x_recon_logs_without_commit():
    host = _LoggingHost(global_step=30)
    host._adapter = MagicMock()
    B, L, T = 2, 1, 6
    host._plot_and_log_x_recon(
        pred_neural=torch.randn(B, L, T),
        pred_x_recon=torch.randn(B, L, T - 1),
        true_x_recon=torch.randn(B, L, T - 1),
        true_neural=torch.randn(B, L, T),
        source_layer=torch.zeros(B, 1, dtype=torch.long),
        num_sources=torch.ones(B, dtype=torch.long),
    )
    host._adapter.log.assert_called_once()
    (payload,), call_kwargs = host._adapter.log.call_args
    assert "media/x_recon" in payload
    assert call_kwargs.get("commit") is False


# -----------------------------
# on_after_backward
# -----------------------------


# `on_after_backward` no longer self-gates on step/cadence -- _shared_step decides whether
# this is a logging step and, if so, sets `_pending_train_log` to the dict it built; these
# tests set that attribute directly to simulate what _shared_step would have produced,
# rather than relying on on_after_backward to reconstruct that decision itself. This is what
# lets a real training step produce exactly one logged row instead of two -- see the
# comment on `_pending_train_log`'s assignment in mich.py::_shared_step.


def test_on_after_backward_noop_when_nothing_pending():
    trainer = _mk_trainer(log_every_n_steps=1)
    host = _LoggingHost(trainer=trainer, global_step=3, heinzle_net=_mk_heinzle_net_double())
    host._adapter = MagicMock()
    # _pending_train_log deliberately left unset -- mirrors a step _shared_step decided
    # wasn't a logging step.
    host.on_after_backward()
    host._adapter.log.assert_not_called()


def test_on_after_backward_logs_pending_dict_without_grad_norms_at_step_zero():
    trainer = _mk_trainer(log_every_n_steps=1)
    host = _LoggingHost(trainer=trainer, global_step=0, heinzle_net=_mk_heinzle_net_double())
    host._adapter = MagicMock()
    host._pending_train_log = {"global_step": 0, "train/loss/total": 1.23}
    host.on_after_backward()

    host._adapter.log.assert_called_once()
    (log_dict,), kwargs = host._adapter.log.call_args
    assert "step" not in kwargs
    assert log_dict["train/loss/total"] == 1.23
    assert not any(k.startswith("gradients/") for k in log_dict)


def test_on_after_backward_noop_when_no_adapter():
    trainer = _mk_trainer(log_every_n_steps=5, is_global_zero=True)
    net = _mk_heinzle_net_double()
    _set_grad(net.spatial_decoder.time_film.linear)
    host = _LoggingHost(trainer=trainer, global_step=10, heinzle_net=net)
    host._pending_train_log = {"global_step": 10}
    host.on_after_backward()  # must not raise despite gradients being present


def test_on_after_backward_clears_pending_even_when_no_adapter():
    trainer = _mk_trainer(log_every_n_steps=5, is_global_zero=True)
    host = _LoggingHost(trainer=trainer, global_step=10, heinzle_net=_mk_heinzle_net_double())
    host._pending_train_log = {"global_step": 10}
    host.on_after_backward()
    assert host._pending_train_log is None


def test_on_after_backward_logs_grad_norms_when_adapter_active():
    trainer = _mk_trainer(log_every_n_steps=5, is_global_zero=True)
    net = _mk_heinzle_net_double()
    _set_grad(net.spatial_decoder.time_film.linear, 2.0)
    _set_grad(net.spatial_decoder.time_film.out, 3.0)
    _set_grad(net.spatial_decoder.out_heads[0], 4.0)
    host = _LoggingHost(trainer=trainer, global_step=10, heinzle_net=net)
    host._adapter = MagicMock()
    host._pending_train_log = {"global_step": 10, "train/loss/total": 0.5}
    host.on_after_backward()

    host._adapter.log.assert_called_once()
    (log_dict,), kwargs = host._adapter.log.call_args
    assert "step" not in kwargs
    assert log_dict["global_step"] == 10
    assert log_dict["train/loss/total"] == 0.5
    assert "gradients/film_linear_norm" in log_dict
    assert "gradients/film_out_norm" in log_dict
    assert "gradients/film_grad_norm" in log_dict
    assert "gradients/out_heads_norm" in log_dict


def test_on_after_backward_omits_keys_with_no_gradients():
    trainer = _mk_trainer(log_every_n_steps=1, is_global_zero=True)
    net = _mk_heinzle_net_double()  # no .grad set on anything
    host = _LoggingHost(trainer=trainer, global_step=1, heinzle_net=net)
    host._adapter = MagicMock()
    host._pending_train_log = {"global_step": 1}
    host.on_after_backward()

    host._adapter.log.assert_called_once()
    (log_dict,), _ = host._adapter.log.call_args
    assert "gradients/film_linear_norm" not in log_dict
    assert "gradients/out_heads_norm" not in log_dict


def test_on_after_backward_no_gpu_keys_in_payload():
    trainer = _mk_trainer(log_every_n_steps=1, is_global_zero=True)
    host = _LoggingHost(trainer=trainer, global_step=1, heinzle_net=_mk_heinzle_net_double())
    host._adapter = MagicMock()
    host._pending_train_log = {"global_step": 1, "train/loss/total": 0.5}
    host.on_after_backward()

    host._adapter.log.assert_called_once()
    (log_dict,), _ = host._adapter.log.call_args
    assert not any(k.startswith("gpu/") for k in log_dict)
    assert log_dict["train/loss/total"] == 0.5


# -----------------------------
# on_fit_start
# -----------------------------


def test_on_fit_start_noop_without_supported_logger(monkeypatch):
    make_adapter_mock = MagicMock()
    monkeypatch.setattr(mich_logging, "make_run_adapter", make_adapter_mock)

    trainer = _mk_trainer(is_global_zero=True, logger=None)  # logger=False in real Trainer usage
    host = _LoggingHost(trainer=trainer)
    host.on_fit_start()  # must not raise (this is the Trainer(logger=False) case)

    make_adapter_mock.assert_not_called()
    assert getattr(host, "_adapter", None) is None


def test_on_fit_start_builds_adapter_and_configures_step_metric_with_wandb_logger(monkeypatch):
    fake_adapter = MagicMock()
    make_adapter_mock = MagicMock(return_value=fake_adapter)
    monkeypatch.setattr(mich_logging, "make_run_adapter", make_adapter_mock)
    logger = WandbLogger(name="test-run", project="test-project")
    trainer = _mk_trainer(is_global_zero=True, logger=logger)
    host = _LoggingHost(trainer=trainer, global_rank=0)

    host.on_fit_start()

    make_adapter_mock.assert_called_once_with(trainer, 0)
    assert host._adapter is fake_adapter
    fake_adapter.configure_step_metric.assert_called_once()


def test_on_fit_start_builds_adapter_with_mlflow_logger(monkeypatch, tmp_path):
    fake_adapter = MagicMock()
    make_adapter_mock = MagicMock(return_value=fake_adapter)
    monkeypatch.setattr(mich_logging, "make_run_adapter", make_adapter_mock)
    logger = MLFlowLogger(
        experiment_name="test-experiment", tracking_uri=f"sqlite:///{tmp_path}/mlflow.db"
    )
    trainer = _mk_trainer(is_global_zero=False, logger=logger)
    host = _LoggingHost(trainer=trainer, global_rank=2)

    host.on_fit_start()

    make_adapter_mock.assert_called_once_with(trainer, 2)
    assert host._adapter is fake_adapter
    fake_adapter.configure_step_metric.assert_called_once()


# -----------------------------
# on_fit_end
# -----------------------------


def test_on_fit_end_finishes_adapter_on_non_zero_rank():
    trainer = _mk_trainer(is_global_zero=False)
    host = _LoggingHost(trainer=trainer)
    fake_adapter = MagicMock()
    host._adapter = fake_adapter

    host.on_fit_end()

    fake_adapter.finish.assert_called_once()
    assert host._adapter is None


def test_on_fit_end_noop_on_global_zero_even_with_adapter_set():
    trainer = _mk_trainer(is_global_zero=True)
    host = _LoggingHost(trainer=trainer)
    fake_adapter = MagicMock()
    host._adapter = fake_adapter

    host.on_fit_end()

    fake_adapter.finish.assert_not_called()


def test_on_fit_end_noop_when_no_adapter_set():
    trainer = _mk_trainer(is_global_zero=False)
    host = _LoggingHost(trainer=trainer)
    host.on_fit_end()  # must not raise despite no _adapter attribute
