"""Unit tests for src/mich/utils/run_adapters.py.

Covers the two concrete RunAdapter implementations, the factory dispatch
(make_run_adapter/make_rank_zero_adapter), and the DDP per-rank creation paths
(wandb reinit / mlflow nested child runs) with the collective torch.distributed
call mocked out -- these can't be exercised with a real multi-process group in
a unit test.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import pytest
from mich.utils import run_adapters
from mich.utils.run_adapters import (
    _MlflowRunAdapter,
    _WandbRunAdapter,
    gpu_stats,
    make_rank_zero_adapter,
    make_run_adapter,
    make_standalone_mlflow_adapter,
)
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger


def _mk_trainer(*, logger=None, world_size=1, is_global_zero=True):
    return types.SimpleNamespace(
        logger=logger, world_size=world_size, is_global_zero=is_global_zero
    )


def _mk_initialized_mlflow_logger(
    tmp_path, *, run_id="parent-run-id", experiment_id="exp-1", run_name=None
):
    """A real MLFlowLogger with its lazy `.experiment`/`.run_id`/`.experiment_id`
    already resolved, so touching those properties never hits a real store.
    tracking_uri still points at a real (tmp_path) sqlite file because
    MLFlowLogger.__init__ eagerly creates its MlflowClient, which itself eagerly
    initializes the sqlite store as a side effect."""
    logger = MLFlowLogger(experiment_name="unused", tracking_uri=f"sqlite:///{tmp_path}/unused.db")
    logger._initialized = True
    logger._run_id = run_id
    logger._experiment_id = experiment_id
    logger._run_name = run_name
    return logger


# -----------------------------
# _WandbRunAdapter
# -----------------------------


def test_wandb_adapter_configure_step_metric_defines_global_step():
    run = MagicMock()
    adapter = _WandbRunAdapter(run)
    adapter.configure_step_metric()
    run.define_metric.assert_any_call("global_step")
    run.define_metric.assert_any_call("*", step_metric="global_step")


def test_wandb_adapter_log_wraps_single_figure_as_image(monkeypatch):
    fake_image_cls = MagicMock(side_effect=lambda fig: ("wrapped", fig))
    monkeypatch.setattr("wandb.Image", fake_image_cls)
    run = MagicMock()
    adapter = _WandbRunAdapter(run)
    fig = plt.figure()

    adapter.log({"global_step": 5, "media/x": fig}, commit=False)

    run.log.assert_called_once()
    (payload,), kwargs = run.log.call_args
    assert payload["media/x"] == ("wrapped", fig)
    assert payload["global_step"] == 5
    assert kwargs == {"commit": False}
    plt.close(fig)


def test_wandb_adapter_log_wraps_figure_list(monkeypatch):
    fake_image_cls = MagicMock(side_effect=lambda fig: ("wrapped", fig))
    monkeypatch.setattr("wandb.Image", fake_image_cls)
    run = MagicMock()
    adapter = _WandbRunAdapter(run)
    figs = [plt.figure(), plt.figure()]

    adapter.log({"global_step": 1, "media/x": figs})

    (payload,), _ = run.log.call_args
    assert payload["media/x"] == [("wrapped", f) for f in figs]
    for f in figs:
        plt.close(f)


def test_wandb_adapter_log_passes_scalars_through_unwrapped():
    run = MagicMock()
    adapter = _WandbRunAdapter(run)
    adapter.log({"global_step": 1, "train/loss": 0.5})
    (payload,), _ = run.log.call_args
    assert payload == {"global_step": 1, "train/loss": 0.5}


def test_wandb_adapter_finish_calls_run_finish():
    run = MagicMock()
    _WandbRunAdapter(run).finish()
    run.finish.assert_called_once()


# -----------------------------
# _MlflowRunAdapter
# -----------------------------


def test_mlflow_adapter_configure_step_metric_is_noop():
    client = MagicMock()
    _MlflowRunAdapter(client, "run-1", is_child=False).configure_step_metric()
    client.assert_not_called()


def test_mlflow_adapter_log_batches_scalars_as_one_call():
    client = MagicMock()
    adapter = _MlflowRunAdapter(client, "run-1", is_child=False)
    adapter.log({"global_step": 7, "train/loss": 0.5, "train/acc": 0.9})

    client.log_batch.assert_called_once()
    _, kwargs = client.log_batch.call_args
    assert kwargs["run_id"] == "run-1"
    logged = {m.key: (m.value, m.step) for m in kwargs["metrics"]}
    assert logged["train/loss"] == (0.5, 7)
    assert logged["train/acc"] == (0.9, 7)
    # global_step is also logged as its own metric (mirrors wandb, where it's a
    # real history column too, not just an implicit x-axis).
    assert logged["global_step"] == (7.0, 7)


def test_mlflow_adapter_log_sends_each_figure_via_log_figure():
    client = MagicMock()
    adapter = _MlflowRunAdapter(client, "run-1", is_child=False)
    figs = [plt.figure(), plt.figure()]

    adapter.log({"global_step": 3, "media/x": figs})

    assert client.log_figure.call_count == 2
    for i, call in enumerate(client.log_figure.call_args_list):
        args, kwargs = call
        assert args[0] == "run-1"
        assert args[1] is figs[i]
        assert kwargs["artifact_file"] == f"media/x/step_{3:08d}_{i}.png"
    for f in figs:
        plt.close(f)


def test_mlflow_adapter_log_ignores_commit_kwarg():
    client = MagicMock()
    adapter = _MlflowRunAdapter(client, "run-1", is_child=False)
    adapter.log({"global_step": 1, "x": 1.0}, commit=True)
    adapter.log({"global_step": 1, "x": 1.0}, commit=False)
    assert client.log_batch.call_count == 2  # both calls succeeded identically


def test_mlflow_adapter_finish_terminates_child_run_only():
    client = MagicMock()
    _MlflowRunAdapter(client, "child-run", is_child=True).finish()
    client.set_terminated.assert_called_once_with("child-run", "FINISHED")


def test_mlflow_adapter_finish_noop_for_parent_run():
    client = MagicMock()
    _MlflowRunAdapter(client, "parent-run", is_child=False).finish()
    client.set_terminated.assert_not_called()


# -----------------------------
# make_run_adapter dispatch
# -----------------------------


def test_make_run_adapter_returns_none_for_unsupported_logger():
    trainer = _mk_trainer(logger=object())
    assert make_run_adapter(trainer, global_rank=0) is None


def test_make_run_adapter_dispatches_to_wandb(monkeypatch):
    fake_run = MagicMock()
    monkeypatch.setattr("wandb.run", fake_run)
    logger = WandbLogger(name="test-run", project="test-project")
    trainer = _mk_trainer(logger=logger)

    adapter = make_run_adapter(trainer, global_rank=0)

    assert isinstance(adapter, _WandbRunAdapter)
    assert adapter._run is fake_run


def test_make_run_adapter_dispatches_to_mlflow(tmp_path):
    logger = _mk_initialized_mlflow_logger(tmp_path)
    trainer = _mk_trainer(logger=logger, world_size=1)

    adapter = make_run_adapter(trainer, global_rank=0)

    assert isinstance(adapter, _MlflowRunAdapter)
    assert adapter._run_id == "parent-run-id"
    assert adapter._is_child is False


# -----------------------------
# _make_wandb_adapter (rank 0 vs non-zero rank)
# -----------------------------


def test_wandb_rank_zero_wraps_existing_wandb_run(monkeypatch):
    fake_run = MagicMock()
    monkeypatch.setattr("wandb.run", fake_run)
    logger = WandbLogger(name="test-run", project="test-project")
    trainer = _mk_trainer(logger=logger)

    adapter = make_run_adapter(trainer, global_rank=0)
    assert adapter._run is fake_run


def test_wandb_non_zero_rank_inits_its_own_reinit_run(monkeypatch):
    fake_rank_run = MagicMock()
    init_mock = MagicMock(return_value=fake_rank_run)
    monkeypatch.setattr("wandb.init", init_mock)
    logger = WandbLogger(name="test-run", project="test-project")
    trainer = _mk_trainer(logger=logger)

    adapter = make_run_adapter(trainer, global_rank=2)

    init_mock.assert_called_once()
    call_kwargs = init_mock.call_args.kwargs
    assert call_kwargs["name"] == "test-run: rank 2"
    assert call_kwargs["reinit"] is True
    assert adapter._run is fake_rank_run


# -----------------------------
# _make_mlflow_adapter (rank 0 vs non-zero rank, incl. DDP broadcast)
# -----------------------------


def test_mlflow_rank_zero_single_process_skips_broadcast(monkeypatch, tmp_path):
    broadcast_mock = MagicMock()
    monkeypatch.setattr(run_adapters.dist, "broadcast_object_list", broadcast_mock)
    logger = _mk_initialized_mlflow_logger(tmp_path, run_id="parent-id", experiment_id="exp-1")
    trainer = _mk_trainer(logger=logger, world_size=1)

    adapter = make_run_adapter(trainer, global_rank=0)

    broadcast_mock.assert_not_called()
    assert adapter._run_id == "parent-id"
    assert adapter._is_child is False


def test_mlflow_rank_zero_ddp_broadcasts_parent_run_id(monkeypatch, tmp_path):
    def _fake_broadcast(obj, src=0):
        assert obj == ["parent-id", "exp-1"]

    monkeypatch.setattr(run_adapters.dist, "broadcast_object_list", _fake_broadcast)
    logger = _mk_initialized_mlflow_logger(tmp_path, run_id="parent-id", experiment_id="exp-1")
    trainer = _mk_trainer(logger=logger, world_size=2)

    adapter = make_run_adapter(trainer, global_rank=0)

    assert adapter._run_id == "parent-id"
    assert adapter._is_child is False


def test_mlflow_non_zero_rank_creates_tagged_child_run(monkeypatch, tmp_path):
    def _fake_broadcast(obj, src=0):
        obj[0] = "parent-id"
        obj[1] = "exp-1"

    monkeypatch.setattr(run_adapters.dist, "broadcast_object_list", _fake_broadcast)

    fake_client = MagicMock()
    fake_client.create_run.return_value = types.SimpleNamespace(
        info=types.SimpleNamespace(run_id="child-run-id")
    )
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda tracking_uri=None: fake_client)

    logger = _mk_initialized_mlflow_logger(
        tmp_path, run_id="parent-id", experiment_id="exp-1", run_name="net: rank 0"
    )
    trainer = _mk_trainer(logger=logger, world_size=2)

    adapter = make_run_adapter(trainer, global_rank=1)

    fake_client.create_run.assert_called_once()
    _, kwargs = fake_client.create_run.call_args
    assert kwargs["experiment_id"] == "exp-1"
    assert kwargs["run_name"] == "net: rank 1"
    from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

    assert kwargs["tags"] == {MLFLOW_PARENT_RUN_ID: "parent-id"}
    assert adapter._run_id == "child-run-id"
    assert adapter._is_child is True


# -----------------------------
# make_rank_zero_adapter
# -----------------------------


def test_rank_zero_adapter_none_on_non_zero_rank():
    trainer = _mk_trainer(logger=WandbLogger(name="n", project="p"), is_global_zero=False)
    assert make_rank_zero_adapter(trainer) is None


def test_rank_zero_adapter_none_when_wandb_run_not_active(monkeypatch):
    monkeypatch.setattr("wandb.run", None)
    trainer = _mk_trainer(logger=WandbLogger(name="n", project="p"), is_global_zero=True)
    assert make_rank_zero_adapter(trainer) is None


def test_rank_zero_adapter_wraps_active_wandb_run(monkeypatch):
    fake_run = MagicMock()
    monkeypatch.setattr("wandb.run", fake_run)
    trainer = _mk_trainer(logger=WandbLogger(name="n", project="p"), is_global_zero=True)
    adapter = make_rank_zero_adapter(trainer)
    assert isinstance(adapter, _WandbRunAdapter)
    assert adapter._run is fake_run


def test_rank_zero_adapter_builds_mlflow_adapter_directly(tmp_path):
    logger = _mk_initialized_mlflow_logger(tmp_path, run_id="run-x")
    trainer = _mk_trainer(logger=logger, is_global_zero=True)
    adapter = make_rank_zero_adapter(trainer)
    assert isinstance(adapter, _MlflowRunAdapter)
    assert adapter._run_id == "run-x"
    assert adapter._is_child is False


def test_rank_zero_adapter_none_for_unsupported_logger():
    trainer = _mk_trainer(logger=object(), is_global_zero=True)
    assert make_rank_zero_adapter(trainer) is None


# -----------------------------
# make_standalone_mlflow_adapter
# -----------------------------


def test_standalone_adapter_creates_run_with_tags_and_flattened_params(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    adapter = make_standalone_mlflow_adapter(
        tracking_uri=uri,
        experiment_name="new-exp",
        run_name="eval/foo",
        tags={"kind": "eval"},
        params={"a": 1, "nested": {"b": 2}},
    )

    assert isinstance(adapter, _MlflowRunAdapter)
    assert adapter._is_child is False

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(adapter._run_id)
    assert run.info.run_name == "eval/foo"
    assert run.data.tags["kind"] == "eval"
    assert run.data.params["a"] == "1"
    assert run.data.params["nested/b"] == "2"  # flattened, '/' delimiter


def test_standalone_adapter_reuses_existing_experiment_across_calls(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    first = make_standalone_mlflow_adapter(tracking_uri=uri, experiment_name="exp", run_name="run1")
    second = make_standalone_mlflow_adapter(
        tracking_uri=uri, experiment_name="exp", run_name="run2"
    )

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    run1 = client.get_run(first._run_id)
    run2 = client.get_run(second._run_id)
    assert run1.info.experiment_id == run2.info.experiment_id
    assert run1.info.run_id != run2.info.run_id


# -----------------------------
# log_artifact / describe
# -----------------------------


def test_wandb_adapter_log_artifact_wraps_as_video(monkeypatch, tmp_path):
    fake_video_cls = MagicMock(side_effect=lambda path, format=None: ("video", path, format))
    monkeypatch.setattr("wandb.Video", fake_video_cls)
    run = MagicMock()
    adapter = _WandbRunAdapter(run)
    gif_path = tmp_path / "evolution.gif"
    gif_path.write_bytes(b"fake")

    adapter.log_artifact(str(gif_path), "media/gif/evolution")

    run.log.assert_called_once_with({"media/gif/evolution": ("video", str(gif_path), "gif")})


def test_wandb_adapter_describe_includes_run_url():
    run = MagicMock()
    run.url = "https://wandb.ai/x/y/runs/z"
    assert _WandbRunAdapter(run).describe() == f"W&B run: {run.url}"


def test_mlflow_adapter_log_artifact_delegates_to_client(tmp_path):
    client = MagicMock()
    adapter = _MlflowRunAdapter(client, "run-1", is_child=False)
    gif_path = tmp_path / "evolution.gif"

    adapter.log_artifact(str(gif_path), "media/gif/evolution")

    client.log_artifact.assert_called_once_with(
        "run-1", str(gif_path), artifact_path="media/gif/evolution"
    )


def test_mlflow_adapter_describe_includes_run_id_and_uri():
    client = MagicMock()
    client.tracking_uri = "sqlite:///x.db"
    adapter = _MlflowRunAdapter(client, "run-1", is_child=False)
    desc = adapter.describe()
    assert "run-1" in desc
    assert "sqlite:///x.db" in desc


# -----------------------------
# gpu_stats
# -----------------------------


def test_gpu_stats_empty_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(run_adapters.torch.cuda, "is_available", lambda: False)
    assert gpu_stats() == {}


def test_gpu_stats_empty_when_pynvml_unavailable(monkeypatch):
    monkeypatch.setattr(run_adapters.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run_adapters, "_ensure_pynvml", lambda: False)
    assert gpu_stats() == {}


def test_gpu_stats_reports_utilization_and_memory(monkeypatch):
    monkeypatch.setattr(run_adapters.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run_adapters, "_ensure_pynvml", lambda: True)
    monkeypatch.setattr(run_adapters.torch.cuda, "current_device", lambda: 0)

    fake_handle = object()
    monkeypatch.setattr(
        "pynvml.nvmlDeviceGetHandleByIndex", lambda idx: fake_handle if idx == 0 else None
    )
    monkeypatch.setattr(
        "pynvml.nvmlDeviceGetUtilizationRates",
        lambda handle: types.SimpleNamespace(gpu=42),
    )
    monkeypatch.setattr(
        "pynvml.nvmlDeviceGetMemoryInfo",
        lambda handle: types.SimpleNamespace(used=1024**2 * 512, total=1024**2 * 2048),
    )

    stats = gpu_stats()

    assert stats == {
        "gpu/utilization_pct": 42.0,
        "gpu/memory_used_mb": 512.0,
        "gpu/memory_total_mb": 2048.0,
    }


def test_gpu_stats_empty_on_nvml_error(monkeypatch):
    monkeypatch.setattr(run_adapters.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run_adapters, "_ensure_pynvml", lambda: True)

    def _raise(idx):
        raise RuntimeError("NVML error")

    monkeypatch.setattr("pynvml.nvmlDeviceGetHandleByIndex", _raise)
    assert gpu_stats() == {}


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
