"""Unit tests for MICHLoggingMixin (src/mich/models/mich_logging.py)."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest
import torch
from pytorch_lightning.loggers import WandbLogger

import wandb
from mich.models.mich_logging import MICHLoggingMixin


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


def test_plot_and_log_predictions_noop_when_no_wandb_run(monkeypatch):
    monkeypatch.setattr(wandb, "run", None)
    host = _LoggingHost(global_step=10)
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


def test_plot_and_log_predictions_logs_one_image_per_sample(monkeypatch):
    fake_run = MagicMock()
    monkeypatch.setattr(wandb, "run", fake_run)
    host = _LoggingHost(global_step=10)
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
    fake_run.log.assert_called_once()
    (payload,), kwargs = fake_run.log.call_args
    assert payload["global_step"] == 10
    assert len(payload["media/predictions"]) == B
    assert kwargs.get("commit") is False


def test_plot_and_log_latents_without_drain_logs_and_commits(monkeypatch):
    fake_run = MagicMock()
    monkeypatch.setattr(wandb, "run", fake_run)
    host = _LoggingHost(global_step=20)
    B, L, T = 2, 2, 5
    kwargs = {
        k: torch.randn(B, L, T)
        for k in ("pred_s", "true_s", "pred_f", "true_f", "pred_v", "true_v", "pred_q", "true_q")
    }
    host._plot_and_log_latents(**kwargs)
    fake_run.log.assert_called_once()
    (payload,), call_kwargs = fake_run.log.call_args
    assert len(payload["media/latents"]) == B
    assert call_kwargs.get("commit") is True


def test_plot_and_log_latents_with_drain_branch_runs(monkeypatch):
    fake_run = MagicMock()
    monkeypatch.setattr(wandb, "run", fake_run)
    host = _LoggingHost(global_step=20)
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
    fake_run.log.assert_called_once()


def test_plot_and_log_x_recon_logs_without_commit(monkeypatch):
    fake_run = MagicMock()
    monkeypatch.setattr(wandb, "run", fake_run)
    host = _LoggingHost(global_step=30)
    B, L, T = 2, 1, 6
    host._plot_and_log_x_recon(
        pred_neural=torch.randn(B, L, T),
        pred_x_recon=torch.randn(B, L, T - 1),
        true_x_recon=torch.randn(B, L, T - 1),
        true_neural=torch.randn(B, L, T),
        source_layer=torch.zeros(B, 1, dtype=torch.long),
        num_sources=torch.ones(B, dtype=torch.long),
    )
    fake_run.log.assert_called_once()
    (payload,), call_kwargs = fake_run.log.call_args
    assert "media/x_recon" in payload
    assert call_kwargs.get("commit") is False


def test_plot_and_log_predictions_prefers_rank_run_over_wandb_run(monkeypatch):
    """When both _rank_run and wandb.run exist, _rank_run (this rank's own run) wins."""
    global_run = MagicMock()
    monkeypatch.setattr(wandb, "run", global_run)
    host = _LoggingHost(global_step=1)
    rank_run = MagicMock()
    host._rank_run = rank_run

    B, L, T = 1, 1, 4
    host._plot_and_log_predictions(
        pred_bold=torch.randn(B, L, T),
        true_bold=torch.randn(B, L, T),
        pred_neural=torch.randn(B, L, T),
        true_neural=torch.randn(B, L, T),
        source_layer=torch.zeros(B, 1, dtype=torch.long),
        source_pos=torch.zeros(B, 1, 2, dtype=torch.long),
        num_sources=torch.ones(B, dtype=torch.long),
    )
    rank_run.log.assert_called_once()
    global_run.log.assert_not_called()


# -----------------------------
# on_after_backward
# -----------------------------


# `on_after_backward` no longer self-gates on step/cadence -- _shared_step decides whether
# this is a logging step and, if so, sets `_pending_train_log` to the dict it built; these
# tests set that attribute directly to simulate what _shared_step would have produced,
# rather than relying on on_after_backward to reconstruct that decision itself. This is what
# lets a real training step produce exactly one wandb history row instead of two -- see the
# comment on `_pending_train_log`'s assignment in mich.py::_shared_step.


def test_on_after_backward_noop_when_nothing_pending(monkeypatch):
    fake_run = MagicMock()
    monkeypatch.setattr(wandb, "run", fake_run)
    trainer = _mk_trainer(log_every_n_steps=1)
    host = _LoggingHost(trainer=trainer, global_step=3, heinzle_net=_mk_heinzle_net_double())
    # _pending_train_log deliberately left unset -- mirrors a step _shared_step decided
    # wasn't a logging step.
    host.on_after_backward()
    fake_run.log.assert_not_called()


def test_on_after_backward_logs_pending_dict_without_grad_norms_at_step_zero(monkeypatch):
    fake_run = MagicMock()
    monkeypatch.setattr(wandb, "run", fake_run)
    trainer = _mk_trainer(log_every_n_steps=1)
    host = _LoggingHost(trainer=trainer, global_step=0, heinzle_net=_mk_heinzle_net_double())
    host._pending_train_log = {"global_step": 0, "train/loss/total": 1.23}
    host.on_after_backward()

    fake_run.log.assert_called_once()
    (log_dict,), kwargs = fake_run.log.call_args
    assert kwargs["step"] == 0
    assert log_dict["train/loss/total"] == 1.23
    assert not any(k.startswith("gradients/") for k in log_dict)


def test_on_after_backward_noop_when_no_active_run(monkeypatch):
    monkeypatch.setattr(wandb, "run", None)
    trainer = _mk_trainer(log_every_n_steps=5, is_global_zero=True)
    net = _mk_heinzle_net_double()
    _set_grad(net.spatial_decoder.time_film.linear)
    host = _LoggingHost(trainer=trainer, global_step=10, heinzle_net=net)
    host._pending_train_log = {"global_step": 10}
    host.on_after_backward()  # must not raise despite gradients being present


def test_on_after_backward_clears_pending_even_when_no_active_run(monkeypatch):
    monkeypatch.setattr(wandb, "run", None)
    trainer = _mk_trainer(log_every_n_steps=5, is_global_zero=True)
    host = _LoggingHost(trainer=trainer, global_step=10, heinzle_net=_mk_heinzle_net_double())
    host._pending_train_log = {"global_step": 10}
    host.on_after_backward()
    assert host._pending_train_log is None


def test_on_after_backward_logs_grad_norms_when_run_active(monkeypatch):
    fake_run = MagicMock()
    monkeypatch.setattr(wandb, "run", fake_run)
    trainer = _mk_trainer(log_every_n_steps=5, is_global_zero=True)
    net = _mk_heinzle_net_double()
    _set_grad(net.spatial_decoder.time_film.linear, 2.0)
    _set_grad(net.spatial_decoder.time_film.out, 3.0)
    _set_grad(net.spatial_decoder.out_heads[0], 4.0)
    host = _LoggingHost(trainer=trainer, global_step=10, heinzle_net=net)
    host._pending_train_log = {"global_step": 10, "train/loss/total": 0.5}
    host.on_after_backward()

    fake_run.log.assert_called_once()
    (log_dict,), kwargs = fake_run.log.call_args
    assert kwargs["step"] == 10
    assert log_dict["global_step"] == 10
    assert log_dict["train/loss/total"] == 0.5
    assert "gradients/film_linear_norm" in log_dict
    assert "gradients/film_out_norm" in log_dict
    assert "gradients/film_grad_norm" in log_dict
    assert "gradients/out_heads_norm" in log_dict


def test_on_after_backward_omits_keys_with_no_gradients(monkeypatch):
    fake_run = MagicMock()
    monkeypatch.setattr(wandb, "run", fake_run)
    trainer = _mk_trainer(log_every_n_steps=1, is_global_zero=True)
    net = _mk_heinzle_net_double()  # no .grad set on anything
    host = _LoggingHost(trainer=trainer, global_step=1, heinzle_net=net)
    host._pending_train_log = {"global_step": 1}
    host.on_after_backward()

    fake_run.log.assert_called_once()
    (log_dict,), _ = fake_run.log.call_args
    assert "gradients/film_linear_norm" not in log_dict
    assert "gradients/out_heads_norm" not in log_dict


def test_on_after_backward_uses_rank_run_when_not_global_zero(monkeypatch):
    monkeypatch.setattr(wandb, "run", MagicMock())  # should be ignored on non-zero rank
    trainer = _mk_trainer(log_every_n_steps=1, is_global_zero=False)
    net = _mk_heinzle_net_double()
    _set_grad(net.spatial_decoder.time_film.linear)
    host = _LoggingHost(trainer=trainer, global_step=1, heinzle_net=net)
    host._pending_train_log = {"global_step": 1}
    rank_run = MagicMock()
    host._rank_run = rank_run
    host.on_after_backward()
    rank_run.log.assert_called_once()


# -----------------------------
# on_fit_start
# -----------------------------


def test_on_fit_start_noop_without_wandb_logger(monkeypatch):
    init_mock = MagicMock()
    monkeypatch.setattr(wandb, "init", init_mock)
    define_metric_mock = MagicMock()
    monkeypatch.setattr(wandb, "define_metric", define_metric_mock)

    trainer = _mk_trainer(is_global_zero=True, logger=None)  # logger=False in real Trainer usage
    host = _LoggingHost(trainer=trainer)
    host.on_fit_start()  # must not raise (this is the Trainer(logger=False) case)

    init_mock.assert_not_called()
    define_metric_mock.assert_not_called()


def test_on_fit_start_global_zero_defines_metrics_with_wandb_logger(monkeypatch):
    define_metric_mock = MagicMock()
    monkeypatch.setattr(wandb, "define_metric", define_metric_mock)
    logger = WandbLogger(name="test-run", project="test-project")
    trainer = _mk_trainer(is_global_zero=True, logger=logger)
    host = _LoggingHost(trainer=trainer)

    host.on_fit_start()

    define_metric_mock.assert_any_call("global_step")
    define_metric_mock.assert_any_call("*", step_metric="global_step")


def test_on_fit_start_non_zero_rank_inits_its_own_run(monkeypatch):
    fake_rank_run = MagicMock()
    init_mock = MagicMock(return_value=fake_rank_run)
    monkeypatch.setattr(wandb, "init", init_mock)
    logger = WandbLogger(name="test-run", project="test-project")
    trainer = _mk_trainer(is_global_zero=False, logger=logger)
    host = _LoggingHost(trainer=trainer, global_rank=2)

    host.on_fit_start()

    init_mock.assert_called_once()
    call_kwargs = init_mock.call_args.kwargs
    assert call_kwargs["name"] == "test-run: rank 2"
    assert call_kwargs["reinit"] is True
    assert host._rank_run is fake_rank_run
    fake_rank_run.define_metric.assert_any_call("global_step")
    fake_rank_run.define_metric.assert_any_call("*", step_metric="global_step")


# -----------------------------
# on_fit_end
# -----------------------------


def test_on_fit_end_finishes_rank_run_on_non_zero_rank():
    trainer = _mk_trainer(is_global_zero=False)
    host = _LoggingHost(trainer=trainer)
    fake_rank_run = MagicMock()
    host._rank_run = fake_rank_run

    host.on_fit_end()

    fake_rank_run.finish.assert_called_once()


def test_on_fit_end_noop_on_global_zero_even_with_rank_run_set():
    trainer = _mk_trainer(is_global_zero=True)
    host = _LoggingHost(trainer=trainer)
    fake_rank_run = MagicMock()
    host._rank_run = fake_rank_run

    host.on_fit_end()

    fake_rank_run.finish.assert_not_called()


def test_on_fit_end_noop_when_no_rank_run_set():
    trainer = _mk_trainer(is_global_zero=False)
    host = _LoggingHost(trainer=trainer)
    host.on_fit_end()  # must not raise despite no _rank_run attribute
