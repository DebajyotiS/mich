"""Backend-agnostic per-rank run logging.

Replaces the direct `wandb`/`mlflow` dispatch (`getattr(self, "_rank_run", None) or
wandb.run`) previously scattered across `mich_logging.py`, `mich.py`, and
`supervised.py`. Those modules only ever touch a `RunAdapter`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import matplotlib.figure
import torch
import torch.distributed as dist
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger

_pynvml_ready = False


def _ensure_pynvml() -> bool:
    global _pynvml_ready
    if _pynvml_ready:
        return True
    try:
        import pynvml

        pynvml.nvmlInit()
        _pynvml_ready = True
    except Exception:
        return False
    return True


def gpu_stats() -> dict[str, float]:
    """Utilization/memory for the GPU this process is actually using (via
    torch.cuda.current_device()), or {} if this process isn't using a GPU or
    pynvml/NVML isn't available. Uses NVML directly rather than torch's own CUDA
    APIs, so it still works even when torch's own CUDA runtime is broken."""
    if not torch.cuda.is_available() or not _ensure_pynvml():
        return {}
    import pynvml

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    except Exception:
        return {}
    return {
        "gpu/utilization_pct": float(util.gpu),
        "gpu/memory_used_mb": mem.used / (1024**2),
        "gpu/memory_total_mb": mem.total / (1024**2),
    }


@runtime_checkable
class RunAdapter(Protocol):
    """Every call site builds a payload dict containing a 'global_step' key."""

    def configure_step_metric(self) -> None: ...
    def log(self, payload: dict[str, Any], commit: bool = True) -> None: ...
    def log_artifact(self, local_path: str, artifact_path: str) -> None: ...
    def describe(self) -> str: ...
    def finish(self) -> None: ...


class _WandbRunAdapter:
    def __init__(self, run) -> None:
        self._run = run

    def configure_step_metric(self) -> None:
        self._run.define_metric("global_step")
        self._run.define_metric("*", step_metric="global_step")

    def log(self, payload: dict[str, Any], commit: bool = True) -> None:
        import wandb

        prepared = {}
        for k, v in payload.items():
            if isinstance(v, matplotlib.figure.Figure):
                prepared[k] = wandb.Image(v)
            elif isinstance(v, list) and v and isinstance(v[0], matplotlib.figure.Figure):
                prepared[k] = [wandb.Image(f) for f in v]
            else:
                prepared[k] = v
        self._run.log(prepared, commit=commit)

    def log_artifact(self, local_path: str, artifact_path: str) -> None:
        import wandb

        suffix = Path(local_path).suffix.lstrip(".") or "mp4"
        self._run.log({artifact_path: wandb.Video(str(local_path), format=suffix)})

    def describe(self) -> str:
        return f"W&B run: {self._run.url}"

    def finish(self) -> None:
        self._run.finish()


class _MlflowRunAdapter:
    def __init__(self, client, run_id: str, is_child: bool) -> None:
        self._client = client
        self._run_id = run_id
        self._is_child = is_child

    def configure_step_metric(self) -> None:
        pass  # Metric.step is a native x-axis; nothing to configure.

    def log(self, payload: dict[str, Any], commit: bool = True) -> None:
        # `commit` has no MLflow analog (no row-merge concept) -- accepted and
        # ignored, kept only so both adapters share one call signature.
        from mlflow.entities import Metric

        step = int(payload.get("global_step", 0))
        timestamp_ms = int(time.time() * 1000)
        metrics, figure_items = [], []
        for k, v in payload.items():
            if isinstance(v, matplotlib.figure.Figure):
                figure_items.append((k, [v]))
            elif isinstance(v, list) and v and isinstance(v[0], matplotlib.figure.Figure):
                figure_items.append((k, v))
            elif isinstance(v, (int, float)):
                metrics.append(Metric(key=k, value=float(v), timestamp=timestamp_ms, step=step))
        if metrics:
            self._client.log_batch(run_id=self._run_id, metrics=metrics)
        for tag, figs in figure_items:
            for i, fig in enumerate(figs):
                self._client.log_figure(
                    self._run_id, fig, artifact_file=f"{tag}/step_{step:08d}_{i}.png"
                )

    def log_artifact(self, local_path: str, artifact_path: str) -> None:
        self._client.log_artifact(self._run_id, str(local_path), artifact_path=artifact_path)

    def describe(self) -> str:
        return f"MLflow run: {self._run_id} (tracking_uri={self._client.tracking_uri})"

    def finish(self) -> None:
        if self._is_child:
            self._client.set_terminated(self._run_id, "FINISHED")


def make_run_adapter(trainer: Trainer, global_rank: int) -> RunAdapter | None:
    """Build the per-rank run adapter for the trainer's active logger, or None if
    the logger isn't one of the backends this adapter layer supports."""
    logger = trainer.logger
    if isinstance(logger, WandbLogger):
        return _make_wandb_adapter(logger, global_rank)
    if isinstance(logger, MLFlowLogger):
        return _make_mlflow_adapter(logger, trainer, global_rank)
    return None


def make_rank_zero_adapter(trainer: Trainer) -> RunAdapter | None:
    """Rank-0-only adapter with no DDP coordination -- for LightningModules that
    don't implement the on_fit_start/on_fit_end per-rank lifecycle. Non-zero ranks
    get None (nothing logs from them), matching pre-existing behavior for those
    modules rather than introducing new cross-rank coordination for them."""
    if not trainer.is_global_zero:
        return None
    logger = trainer.logger
    if isinstance(logger, WandbLogger):
        import wandb

        return _WandbRunAdapter(wandb.run) if wandb.run is not None else None
    if isinstance(logger, MLFlowLogger):
        return _MlflowRunAdapter(logger._mlflow_client, logger.run_id, is_child=False)
    return None


def _make_wandb_adapter(logger: WandbLogger, global_rank: int) -> _WandbRunAdapter:
    import wandb

    if global_rank == 0:
        return _WandbRunAdapter(wandb.run)
    base_name = logger._wandb_init.get("name", logger.name).rsplit(": rank", 1)[0]
    init_kwargs = {**logger._wandb_init, "name": f"{base_name}: rank {global_rank}", "reinit": True}
    return _WandbRunAdapter(wandb.init(**init_kwargs))


def _make_mlflow_adapter(
    logger: MLFlowLogger, trainer: Trainer, global_rank: int
) -> _MlflowRunAdapter:
    from mlflow.tracking import MlflowClient

    world_size = trainer.world_size
    obj = [logger.run_id, logger.experiment_id] if global_rank == 0 else [None, None]
    if world_size > 1:
        dist.broadcast_object_list(obj, src=0)
    parent_run_id, experiment_id = obj

    if global_rank == 0:
        return _MlflowRunAdapter(logger._mlflow_client, parent_run_id, is_child=False)

    from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

    client = MlflowClient(tracking_uri=logger._tracking_uri)
    base_name = (logger._run_name or "run").rsplit(": rank", 1)[0]
    child = client.create_run(
        experiment_id=experiment_id,
        run_name=f"{base_name}: rank {global_rank}",
        tags={MLFLOW_PARENT_RUN_ID: parent_run_id},
    )
    return _MlflowRunAdapter(client, child.info.run_id, is_child=True)


def make_standalone_mlflow_adapter(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    tags: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> _MlflowRunAdapter:
    """Single top-level MLflow run for non-DDP entry points (e.g. `eval_mich.py`) --
    no parent/child nesting needed since these never run under a process group.
    Reuses MLFlowLogger for experiment/run creation and hyperparameter flattening
    rather than re-implementing that against MlflowClient directly."""
    logger = MLFlowLogger(
        experiment_name=experiment_name, run_name=run_name, tracking_uri=tracking_uri, tags=tags
    )
    if params:
        logger.log_hyperparams(params)
    return _MlflowRunAdapter(logger._mlflow_client, logger.run_id, is_child=False)


def make_standalone_wandb_adapter(
    project: str, name: str, entity: str | None = None, config: dict[str, Any] | None = None
) -> _WandbRunAdapter:
    """Single top-level W&B run for non-DDP entry points (e.g. `eval_mich.py`)."""
    import wandb

    run = wandb.init(project=project, entity=entity, name=name, config=config)
    return _WandbRunAdapter(run)
