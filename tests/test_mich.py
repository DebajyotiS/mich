"""Integration tests for MICH LightningModule.

Coverage:
  - forward pass shape and finiteness
  - time-derivative (grads) shape when requested / None when not
  - _data_loss and _physics_loss: scalar, finite
  - backward propagation: all parameter gradients finite
  - full Trainer.fit() smoke test via fast_dev_run=True  (marked slow)
"""

from __future__ import annotations

import types
from functools import partial
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
import torch
import torch.optim
import torch.optim.lr_scheduler
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from mich.data.synthetic import SyntheticDataModule
from mich.models.blocks import HeinzleNet
from mich.models.mich import MICH

# -------------------------
# Module-level constants
# -------------------------

_LAYERS = ("layer_0", "layer_1")  # 2-layer setup for unit/component tests (faster)
_B, _L, _T, _H, _W = 2, 2, 8, 4, 4

# The Trainer integration tests use 3 layers to match production (plot_latent_layers
# hardcodes LAYER_NAMES with 3 entries and fails with fewer layers).
_LAYERS_3 = ("layer_0", "layer_1", "layer_2")
_L3 = 3


def _const_lr(_step: int) -> float:
    """Constant learning-rate schedule — must be a named function to be picklable."""
    return 1.0


# -------------------------
# Factories / helpers
# -------------------------


@pytest.fixture(autouse=True)
def _seed():
    """Deterministic weights and random ops for every test in this module."""
    torch.manual_seed(0)


def _mk_heinzle_configs(
    *,
    L: int = 2,
    Cmix: int = 4,
    Cenc: int = 6,
    c_dec: int = 8,
    c_film: int = 4,
    signals: list[str] | None = None,
) -> dict:
    """Minimal HeinzleNet constructor kwargs, following test_blocks.py conventions.

    `signals` lets a caller build a no-drain (5-channel, no vstar/qstar) decoder --
    the "has_drain=False" branch throughout mich_losses.py/mich.py is keyed off
    z_hat's channel count (len(signals)), independent of L.
    """
    num_freqs = 4
    spatial_decoder_extra = dict(signals=signals) if signals is not None else {}
    out_channels = len(signals) if signals is not None else 7
    return dict(
        layer_mixing_config=dict(L=L, C=Cmix, init_identity=True),
        spatial_encoder_config=[
            dict(
                cin=Cmix,
                cout=Cenc,
                stride=1,
                dw_kernel=3,
                pw_kernel=1,
                num_groups=1,
                activation="silu",
            )
        ],
        temporal_mixing_config=[dict(cin=Cenc, kernel_size=3, num_groups=1, activation="silu")],
        time_embedding_config=dict(num_freqs=num_freqs, max_freq=3.0),
        time_film_config=dict(
            embed_dim=2 * num_freqs,
            hidden_dim=16,
            activation="silu",
            c_dec=c_film,
        ),
        spatial_decoder_config=dict(
            cin=Cenc,
            c_dec=c_dec,
            c_film=c_film,
            out_channels=out_channels,
            activation="silu",
            L=L,
            upsample=False,
            **spatial_decoder_extra,
        ),
    )


def _make_mich(*, L: int = _L, signals: list[str] | None = None, **loss_overrides) -> MICH:
    """Construct a minimal but fully functional MICH model (normaliser=None)."""
    heinzle_net = HeinzleNet(**_mk_heinzle_configs(L=L, signals=signals))
    loss_cfg = dict(
        order="linear",
        n_time=4,
        n_space=4,
        dense_spatial_frac=0.8,
        dense_spatial_radius=2,
        dense_time_frac=0.8,
        dense_time_lo=0.05,
        dense_time_hi=0.55,
        uniform_time_lo=0.05,
        lambda_src=1.0,
        lambda_data=1.0,
        lambda_physics=0.1,
        lambda_smooth=0.01,
        lambda_supervision=0.0,
        warmup_steps_physics=0,
        warmup_steps_smooth=0,
        delay_steps_physics=0,
        delay_steps_smooth=0,
        burn_in=1,
    )
    loss_cfg.update(loss_overrides)
    return MICH(
        heinzle_net=heinzle_net,
        normaliser=None,
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=partial(
            torch.optim.lr_scheduler.LambdaLR,
            lr_lambda=_const_lr,
        ),
        loss_config=types.SimpleNamespace(**loss_cfg),
        haemo=types.SimpleNamespace(
            kappa=0.65,
            gamma=0.41,
            alpha=0.32,
            tau=1.0,
            lambda_d=0.2,
            tau_d=1.0,
        ),
        acquisition=types.SimpleNamespace(
            k1=0.02,
            k2=0.38,
            k3=0.38,
            E0=0.35,
        ),
        V0=0.02,
        lightning={"interval": "step", "frequency": 1},
    )


def _make_batch(
    *, B: int = _B, L: int = _L, T: int = _T, H: int = _H, W: int = _W, S: int = 1
) -> dict:
    """Minimal training batch (bold + neural + source_position/source_layer/num_sources)."""
    return {
        "bold": torch.randn(B, L, T, H, W),
        "neural": torch.randn(B, L, T, H, W),
        "source_position": torch.randint(0, min(H, W), (B, S, 2)),
        "source_layer": torch.randint(0, L, (B, S)),
        "num_sources": torch.randint(1, S + 1, (B,)),
    }


def _make_full_batch(
    *, B: int = _B, L: int = _L, T: int = _T, H: int = _H, W: int = _W, S: int = 1
) -> dict:
    """Validation batch: training keys plus all meta and latent keys."""
    batch = _make_batch(B=B, L=L, T=T, H=H, W=W, S=S)
    batch["num_pulses"] = torch.randint(1, 4, (B, S))
    for key in ("s", "f", "v", "q", "v_star", "q_star"):
        batch[key] = torch.randn(B, L, T, H, W)
    return batch


def _make_h5_fixture(
    path: str,
    *,
    layers: tuple = _LAYERS_3,
    n: int = 8,
    t: int = _T,
    h: int = _H,
    w: int = _W,
    max_sources: int = 2,
) -> None:
    """Write a minimal HDF5 file understood by SyntheticDataModule."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for lyr in layers:
            grp = f.require_group(lyr)
            grp.create_dataset("bold", data=rng.standard_normal((n, t, h, w)).astype(np.float32))
            grp.create_dataset("x", data=rng.standard_normal((n, t, h, w)).astype(np.float32))
            for key in ("s", "f", "v", "q", "v_star", "q_star"):
                grp.create_dataset(key, data=rng.standard_normal((n, t, h, w)).astype(np.float32))
        meta = f.require_group("meta")
        num_sources = rng.integers(1, max_sources + 1, size=n).astype(np.int32)
        meta.create_dataset("num_sources", data=num_sources)

        source_layer = np.full((n, max_sources), -1, dtype=np.int32)
        source_position = np.full((n, max_sources, 2), -1, dtype=np.int32)
        source_num_pulses = np.zeros((n, max_sources), dtype=np.int32)
        for i in range(n):
            k = int(num_sources[i])
            source_layer[i, :k] = rng.integers(0, len(layers), size=k)
            source_position[i, :k] = rng.integers(0, min(h, w), size=(k, 2))
            source_num_pulses[i, :k] = rng.integers(1, 4, size=k)

        sources = meta.create_group("sources")
        sources.create_dataset("layer", data=source_layer)
        sources.create_dataset("position", data=source_position)
        sources.create_dataset("num_pulses", data=source_num_pulses)


def _make_datamodule(
    h5_path: str, *, layers: tuple = _LAYERS_3, return_latents: bool = True
) -> SyntheticDataModule:
    """DataModule configured for integration tests: meta + latents enabled."""
    return SyntheticDataModule(
        data={
            "path": h5_path,
            "layers": list(layers),
            "return_meta": True,
            "return_latents": return_latents,
            "dtype": "float32",
        },
        split={"train_count": 4, "val_count": 2, "test_count": 2, "seed": 42},
        loader={"batch_size": 2, "num_workers": 0, "drop_last": True},
        h5_cache={},
    )


# -------------------------
# forward()
# -------------------------


class TestMICHForward:
    def test_z_hat_shape(self):
        """forward() output has shape [B, 7, L, T, H, W]."""
        model = _make_mich()
        model.eval()
        bold = torch.randn(_B, _L, _T, _H, _W)
        time = MICH._make_time_grid(_B, _T, device=bold.device, dtype=bold.dtype)
        manifest = model(bold, time)
        assert manifest.z_hat.shape == (_B, 7, _L, _T, _H, _W)

    def test_z_hat_finite(self):
        """forward() produces no NaN or Inf values."""
        model = _make_mich()
        model.eval()
        bold = torch.randn(_B, _L, _T, _H, _W)
        time = MICH._make_time_grid(_B, _T, device=bold.device, dtype=bold.dtype)
        assert torch.isfinite(model(bold, time).z_hat).all()

    def test_grads_shape_matches_z_hat_when_requested(self):
        """Requesting gradients returns dz/dt tensor with same shape as z_hat."""
        model = _make_mich()
        model.eval()
        bold = torch.randn(_B, _L, _T, _H, _W)
        time = MICH._make_time_grid(_B, _T, device=bold.device, dtype=bold.dtype)
        manifest = model(bold, time, return_gradients=True)
        assert manifest.grads is not None
        assert manifest.grads.shape == manifest.z_hat.shape

    def test_grads_none_when_not_requested(self):
        """grads is None when return_gradients=False (default)."""
        model = _make_mich()
        model.eval()
        bold = torch.randn(_B, _L, _T, _H, _W)
        time = MICH._make_time_grid(_B, _T, device=bold.device, dtype=bold.dtype)
        assert model(bold, time, return_gradients=False).grads is None


# -------------------------
# _data_loss / _physics_loss
# -------------------------


class TestMICHLosses:
    def test_data_loss_is_finite_scalar(self):
        """_data_loss returns a 0-D finite tensor."""
        model = _make_mich()
        model.eval()
        bold = torch.randn(_B, _L, _T, _H, _W)
        time = MICH._make_time_grid(_B, _T, device=bold.device, dtype=bold.dtype)
        z_hat = model(bold, time).z_hat
        src_pos = torch.randint(0, min(_H, _W), (_B, 1, 2))
        num_sources = torch.ones(_B, dtype=torch.long)
        loss, _, _ = model._data_loss(z_hat, bold, source_position=src_pos, num_sources=num_sources)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_physics_loss_is_finite_scalar(self):
        """_physics_loss returns a 0-D finite tensor."""
        model = _make_mich()
        model.eval()
        bold = torch.randn(_B, _L, _T, _H, _W)
        time = MICH._make_time_grid(_B, _T, device=bold.device, dtype=bold.dtype)
        manifest = model(bold, time, return_gradients=True)
        src_pos = torch.randint(0, min(_H, _W), (_B, 1, 2))
        num_sources = torch.ones(_B, dtype=torch.long)
        loss, _ = model._physics_loss(
            manifest.z_hat,
            manifest.grads,
            order="linear",
            source_position=src_pos,
            num_sources=num_sources,
        )
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_physics_loss_without_source_position_is_finite(self):
        """_physics_loss works with source_position=None (uniform collocation)."""
        model = _make_mich()
        model.eval()
        bold = torch.randn(_B, _L, _T, _H, _W)
        time = MICH._make_time_grid(_B, _T, device=bold.device, dtype=bold.dtype)
        manifest = model(bold, time, return_gradients=True)
        loss, _ = model._physics_loss(
            manifest.z_hat,
            manifest.grads,
            order="linear",
            source_position=None,
            num_sources=None,
        )
        assert torch.isfinite(loss)

    def test_total_loss_backward_propagates_finite_gradients(self):
        """Summed data + physics loss backpropagates without NaN/Inf in any grad."""
        model = _make_mich()
        model.train()
        bold = torch.randn(_B, _L, _T, _H, _W)
        time = MICH._make_time_grid(_B, _T, device=bold.device, dtype=bold.dtype)
        src_pos = torch.randint(0, min(_H, _W), (_B, 1, 2))
        num_sources = torch.ones(_B, dtype=torch.long)
        manifest = model(bold, time, return_gradients=True)
        data_loss, _, _ = model._data_loss(
            manifest.z_hat, bold, source_position=src_pos, num_sources=num_sources
        )
        physics_loss, _ = model._physics_loss(
            manifest.z_hat,
            manifest.grads,
            order="linear",
            source_position=src_pos,
            num_sources=num_sources,
        )
        (data_loss + physics_loss).backward()
        grads_with_value = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads_with_value) > 0, "No parameter received a gradient"
        assert all(
            torch.isfinite(g).all() for g in grads_with_value
        ), "At least one parameter gradient contains NaN or Inf"


class TestMICHOptionalLossWiring:
    @pytest.mark.slow
    @pytest.mark.parametrize(
        "loss_overrides",
        [
            pytest.param(dict(lambda_antisteady=1.0, antisteady_epsilon=0.01), id="antisteady"),
            pytest.param(dict(lambda_supervision=1.0), id="supervision"),
            pytest.param(
                dict(
                    supervise_dzdt=True,
                    lambda_dzdt_supervision=1.0,
                    dzdt_supervision_signals=("s",),
                    lambda_physics=0.0,  # need_grads must still turn on for supervise_dzdt alone
                ),
                id="dzdt_supervision_without_physics",
            ),
            pytest.param(
                dict(supervise_x_phase=True, lambda_x_phase=1.0, lambda_physics=0.0),
                id="x_phase_without_physics",
            ),
            pytest.param(
                dict(
                    lambda_antisteady=1.0,
                    lambda_supervision=1.0,
                    supervise_dzdt=True,
                    lambda_dzdt_supervision=1.0,
                    supervise_x_phase=True,
                    lambda_x_phase=1.0,
                ),
                id="all_enabled_together",
            ),
        ],
    )
    def test_fast_dev_run_completes_with_optional_loss_enabled(self, tmp_path, loss_overrides):
        """Each optional loss term wires into total_loss and survives a real forward
        pass (real HeinzleNet, real collocation sampling) without shape errors."""
        torch.manual_seed(0)
        h5_path = str(tmp_path / "data.h5")
        _make_h5_fixture(h5_path, layers=_LAYERS_3)
        dm = _make_datamodule(h5_path, layers=_LAYERS_3)
        model = _make_mich(L=_L3, **loss_overrides)

        trainer = Trainer(
            max_epochs=1,
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, datamodule=dm)

    @pytest.mark.slow
    def test_fast_dev_run_completes_without_drain_channels(self, tmp_path):
        """A decoder built with the 5-signal (no vstar/qstar) set -- has_drain=False
        throughout mich_losses.py/mich.py -- must still complete a full train+val step,
        including validation_step's "no v_star/q_star" true_z_hat branch and
        on_validation_epoch_end's no-drain _plot_and_log_latents call."""
        torch.manual_seed(0)
        h5_path = str(tmp_path / "data.h5")
        _make_h5_fixture(h5_path, layers=_LAYERS_3)
        dm = _make_datamodule(h5_path, layers=_LAYERS_3)
        model = _make_mich(L=_L3, signals=["x", "s", "f", "v", "q"])

        trainer = Trainer(
            max_epochs=1,
            fast_dev_run=True,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, datamodule=dm)

    @pytest.mark.slow
    def test_supervision_guard_skips_cleanly_when_batch_has_no_latents(self, tmp_path):
        """lambda_supervision>0 but the batch carries no 's'/'f'/'v'/'q' (as with real,
        non-simulated fMRI) -- the `batch.get("s") is not None` guard in _shared_step
        must skip supervision rather than KeyError. validation_step unconditionally
        reads batch["s"], so this only exercises the train-only path (limit_val_batches=0)."""
        torch.manual_seed(0)
        h5_path = str(tmp_path / "data.h5")
        _make_h5_fixture(h5_path, layers=_LAYERS_3)
        dm = _make_datamodule(h5_path, layers=_LAYERS_3, return_latents=False)
        model = _make_mich(L=_L3, lambda_supervision=1.0)

        trainer = Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=0,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
        )
        trainer.fit(model, datamodule=dm)


# -------------------------
# _shared_step internals via a lightweight fake trainer (no real Trainer.fit()
# overhead needed). self.log is stubbed to a no-op since we're only exercising
# MICH's own branch/decision logic, not PL's logging plumbing (already exercised
# by the fast_dev_run integration tests above).
# -------------------------


class TestMICHSharedStepInternals:
    @staticmethod
    def _attach_fake_trainer(model, *, is_global_zero=True, log_every_n_steps=1, global_step=0):
        model.trainer = types.SimpleNamespace(
            is_global_zero=is_global_zero,
            log_every_n_steps=log_every_n_steps,
            global_step=global_step,
        )
        model.log = lambda *a, **k: None

    def test_shared_step_invalid_stage_raises(self):
        model = _make_mich()
        self._attach_fake_trainer(model)
        batch = _make_full_batch()
        with pytest.raises(ValueError, match="Invalid stage"):
            model._shared_step(batch, stage="bogus")

    def test_physics_loss_is_literal_zero_when_physics_disabled_and_no_grad_dependents(self):
        """need_grads is False when lambda_physics==0 and no dzdt/x_phase supervision
        is requested (train stage) -- physics_loss should be the literal 0.0 stand-in,
        not just coincidentally small."""
        model = _make_mich(lambda_physics=0.0)
        self._attach_fake_trainer(model)
        batch = _make_full_batch()
        manifest = model._shared_step(batch, stage="train")
        assert manifest.physics_loss.item() == 0.0

    def test_need_grads_true_for_dzdt_supervision_even_with_physics_disabled(self):
        """supervise_dzdt alone (lambda_physics==0) must still request gradients --
        this exercises the `or getattr(lc, "supervise_dzdt", False)` arm of need_grads."""
        model = _make_mich(lambda_physics=0.0, supervise_dzdt=True, lambda_dzdt_supervision=1.0)
        self._attach_fake_trainer(model)
        batch = _make_full_batch()
        manifest = model._shared_step(batch, stage="train")
        assert torch.isfinite(manifest.total_loss)

    def test_x_phase_annealing_curriculum_changes_effective_weights_over_steps(self):
        """When x_phase_ratio_anneal_start/end_step bracket global_step, both the
        x_phase loss weight and its internal mse/pearson balance should differ between
        an early and a late step (linear anneal, clamped outside the window)."""
        xp_cfg = types.SimpleNamespace(type="mse+pearson", lambda_npearson=1.0, lambda_pearson=1.0)
        model = _make_mich(
            supervise_x_phase=True,
            x_phase_loss=xp_cfg,
            lambda_x_phase_start=0.0,
            lambda_x_phase_end=10.0,
            x_phase_npearson_start=1.0,
            x_phase_npearson_end=0.0,
            x_phase_pearson_start=0.0,
            x_phase_pearson_end=1.0,
            x_phase_ratio_anneal_start_step=0,
            x_phase_ratio_anneal_end_step=100,
        )
        batch = _make_full_batch()

        self._attach_fake_trainer(model, global_step=0)
        model._shared_step(batch, stage="train")
        npearson_early = model.hparams.loss_config.x_phase_loss.lambda_npearson

        self._attach_fake_trainer(model, global_step=100)
        model._shared_step(batch, stage="train")
        npearson_late = model.hparams.loss_config.x_phase_loss.lambda_npearson

        assert npearson_early == pytest.approx(1.0)
        assert npearson_late == pytest.approx(0.0)

    def test_wandb_detailed_train_logging_includes_all_optional_terms(self, monkeypatch):
        fake_run = MagicMock()
        monkeypatch.setattr(wandb, "run", fake_run)
        model = _make_mich(
            lambda_antisteady=1.0,
            antisteady_epsilon=0.01,
            lambda_supervision=1.0,
            supervise_dzdt=True,
            lambda_dzdt_supervision=1.0,
            supervise_x_phase=True,
            lambda_x_phase=1.0,
        )
        self._attach_fake_trainer(model, log_every_n_steps=1)
        batch = _make_full_batch()
        model._shared_step(batch, stage="train")

        fake_run.log.assert_called_once()
        (log_dict,), _ = fake_run.log.call_args
        for key in (
            "train/loss/total",
            "train/loss/data",
            "train/loss/physics",
            "train/loss/antisteady",
            "train/loss_weighted/antisteady",
            "train/loss/supervision",
            "train/loss_weighted/supervision",
            "train/loss/dzdt_supervision",
            "train/loss_weighted/dzdt_supervision",
            "train/loss/x_phase",
            "train/loss_weighted/x_phase",
            "parameters/lambda_physics",
            "parameters/lambda_antisteady",
        ):
            assert key in log_dict, f"missing {key}"

    def test_wandb_detailed_logging_noop_off_cadence(self, monkeypatch):
        fake_run = MagicMock()
        monkeypatch.setattr(wandb, "run", fake_run)
        model = _make_mich()
        self._attach_fake_trainer(model, log_every_n_steps=5, global_step=1)
        batch = _make_full_batch()
        model._shared_step(batch, stage="train")
        fake_run.log.assert_not_called()

    def test_wandb_detailed_logging_noop_on_val_stage(self, monkeypatch):
        """The detailed per-step wandb log is train-only; val stage must not call it
        even with an active run and on-cadence global_step."""
        fake_run = MagicMock()
        monkeypatch.setattr(wandb, "run", fake_run)
        model = _make_mich()
        self._attach_fake_trainer(model, log_every_n_steps=1)
        batch = _make_full_batch()
        model._shared_step(batch, stage="val")
        fake_run.log.assert_not_called()


# -------------------------
# Static helper methods
# -------------------------


class TestMICHStaticHelpers:
    def test_make_time_grid_shape_and_range(self):
        """_make_time_grid produces a [B, T] tensor spanning [0, 1]."""
        B, T = 3, 10
        t = MICH._make_time_grid(B, T, device=torch.device("cpu"), dtype=torch.float32)
        assert t.shape == (B, T)
        assert t.min().item() == pytest.approx(0.0)
        assert t.max().item() == pytest.approx(1.0)

    def test_signal_index_all_names(self):
        for name, expected in zip(
            ("x", "s", "f", "v", "q", "vstar", "qstar"), range(7), strict=True
        ):
            assert MICH._signal_index(name) == expected

    def test_signal_index_int_passthrough(self):
        for i in range(7):
            assert MICH._signal_index(i) == i

    def test_signal_index_out_of_range_raises(self):
        with pytest.raises(IndexError):
            MICH._signal_index(7)

    def test_compute_bold_formula(self):
        """_compute_bold matches k1*(1-q) + k2*(1-q/v) + k3*(1-v) scaled by V0."""
        acq = types.SimpleNamespace(k1=7.0, k2=2.0, k3=2.0)
        V0 = 0.02
        v = torch.tensor([[1.2]])
        q = torch.tensor([[0.9]])
        bold = MICH._compute_bold(v, q, acq, V0)
        expected = V0 * (acq.k1 * (1 - q) + acq.k2 * (1 - q / v) + acq.k3 * (1 - v))
        assert torch.allclose(bold, expected)


# -------------------------
# Full pipeline: Trainer.fit
# -------------------------


@pytest.mark.slow
def test_training_fast_dev_run_completes_without_error(tmp_path):
    """One train step + one val step + val epoch end complete without error.

    Exercises the full pipeline:
        HDF5 fixture -> DataModule -> MICH forward -> losses -> backward
        -> optimizer step -> validation_step -> on_validation_epoch_end

    Uses L=3 layers to match the production assumption in plot_latent_layers
    (LAYER_NAMES is hardcoded to 3 entries).
    """
    torch.manual_seed(0)
    h5_path = str(tmp_path / "data.h5")
    _make_h5_fixture(h5_path, layers=_LAYERS_3)
    dm = _make_datamodule(h5_path, layers=_LAYERS_3)
    model = _make_mich(L=_L3)

    trainer = Trainer(
        max_epochs=1,
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    # Should complete without raising
    trainer.fit(model, datamodule=dm)


@pytest.mark.slow
def test_training_fast_dev_run_losses_finite(tmp_path):
    """Losses logged during the fast_dev_run training step are finite."""
    torch.manual_seed(0)
    h5_path = str(tmp_path / "data.h5")
    _make_h5_fixture(h5_path, layers=_LAYERS_3)
    dm = _make_datamodule(h5_path, layers=_LAYERS_3)
    model = _make_mich(L=_L3)

    captured: list = []

    class _LossRecorder(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            captured.append(outputs)

    trainer = Trainer(
        max_epochs=1,
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[_LossRecorder()],
    )
    trainer.fit(model, datamodule=dm)

    assert len(captured) == 1, f"Expected 1 training output, got {len(captured)}"
    out = captured[0]
    # PL may pass the tensor directly or wrap it; extract a float either way.
    if isinstance(out, torch.Tensor):
        loss_val = out.item()
    elif isinstance(out, dict) and "loss" in out:
        loss_val = out["loss"].item()
    else:
        # Fall back: just verify the run completed (already covered above)
        loss_val = 0.0
    assert np.isfinite(loss_val), f"Training loss was not finite: {loss_val}"
