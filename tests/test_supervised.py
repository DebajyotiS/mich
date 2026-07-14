"""Unit/integration tests for SupervisedMICH (src/mich/models/supervised.py).

Coverage:
  - _pearson_loss: correlated / anti-correlated / uncorrelated signals
  - _neural_recovery_metrics: identical signals, delayed signals, shape/keys
  - forward(): normaliser=None passthrough vs. normaliser invoked and its
    output routed into net
  - _shared_step: finite loss; val-stage buffer growth vs. train-stage
    buffers untouched (via a real Trainer.fit(), since self.log() requires
    an attached Trainer)
  - training_step / validation_step: thin-wrapper delegation to _shared_step
  - on_validation_epoch_end: empty-buffer no-op, wandb.run is None path,
    wandb.run mocked path (scalar metrics + image logging + buffer clearing),
    and an empirical check of the bold/true source-position pairing in the
    plotting subset (see TestOnValidationEpochEndSourcePairing docstring)
  - configure_optimizers: returned dict structure
  - full Trainer.fit() smoke test (marked slow)
"""

from __future__ import annotations

import types
from functools import partial
from unittest.mock import MagicMock

import pytest
import torch
import torch.optim
import torch.optim.lr_scheduler
import wandb
from mich.models.blocks import FullySupervisedNet
from mich.models.supervised import SupervisedMICH
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, Dataset

# -------------------------
# Module-level constants
# -------------------------

_B, _L, _T, _H, _W = 2, 2, 8, 4, 4


def _const_lr(_step: int) -> float:
    """Constant learning-rate schedule — must be a named function to be picklable."""
    return 1.0


@pytest.fixture(autouse=True)
def _seed():
    """Deterministic weights and random ops for every test in this module."""
    torch.manual_seed(0)


# -------------------------
# Factories / helpers
# -------------------------


def _mk_supervised_net(*, L: int = _L, Cmix: int = 4, Cenc: int = 6) -> FullySupervisedNet:
    """Minimal FullySupervisedNet constructor kwargs, mirroring test_mich.py's HeinzleNet helper."""
    return FullySupervisedNet(
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
        c_enc=Cenc,
    )


def _make_model(
    *, L: int = _L, net: FullySupervisedNet | None = None, normaliser=None, lambda_pearson=1.0
) -> SupervisedMICH:
    """Construct a minimal but fully functional SupervisedMICH (normaliser=None by default)."""
    if net is None:
        net = _mk_supervised_net(L=L)
    return SupervisedMICH(
        net=net,
        normaliser=normaliser,
        optimizer=partial(torch.optim.Adam, lr=1e-3),
        scheduler=partial(torch.optim.lr_scheduler.LambdaLR, lr_lambda=_const_lr),
        loss_config=types.SimpleNamespace(lambda_pearson=lambda_pearson),
        lightning={"interval": "step", "frequency": 1},
    )


def _make_batch(*, B: int = _B, L: int = _L, T: int = _T, H: int = _H, W: int = _W) -> dict:
    """Minimal batch matching what SupervisedMICH._shared_step reads.

    Note: source_position here is [B, 2] (single source per sample, no
    max_sources dim) — that's what `_shared_step` actually indexes
    (`source_position[:, 0]` / `[:, 1]`), unlike MICH's [B, S, 2] convention.
    """
    return {
        "bold": torch.randn(B, L, T, H, W),
        "neural": torch.randn(B, L, T, H, W),
        "source_position": torch.randint(0, min(H, W), (B, 2)),
    }


class _BatchListDataset(Dataset):
    """Dataset of pre-built batch dicts; used with DataLoader(batch_size=None)."""

    def __init__(self, batches: list[dict]):
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> dict:
        return self.batches[idx]


def _make_loader(batches: list[dict]) -> DataLoader:
    return DataLoader(_BatchListDataset(batches), batch_size=None, shuffle=False)


# -------------------------
# _pearson_loss
# -------------------------


class TestPearsonLoss:
    def test_perfectly_correlated_is_near_zero(self):
        true = torch.randn(4, 16)
        pred = 3.0 * true + 5.0  # affine transform preserves correlation = 1
        loss = SupervisedMICH._pearson_loss(pred, true)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_perfectly_anticorrelated_is_near_two(self):
        true = torch.randn(4, 16)
        pred = -2.0 * true + 1.0
        loss = SupervisedMICH._pearson_loss(pred, true)
        assert loss.item() == pytest.approx(2.0, abs=1e-5)

    def test_uncorrelated_is_finite_and_between_extremes(self):
        torch.manual_seed(1)
        true = torch.randn(8, 64)
        pred = torch.randn(8, 64)
        loss = SupervisedMICH._pearson_loss(pred, true)
        assert torch.isfinite(loss)
        # Independent random signals: correlation should be far from +/-1,
        # i.e. loss should be far from the 0.0 / 2.0 extremes.
        assert 0.5 < loss.item() < 1.5

    def test_returns_scalar(self):
        true = torch.randn(4, 16)
        pred = torch.randn(4, 16)
        loss = SupervisedMICH._pearson_loss(pred, true)
        assert loss.ndim == 0


# -------------------------
# _neural_recovery_metrics
# -------------------------


class TestNeuralRecoveryMetrics:
    def test_identical_signals_perfect_recovery(self):
        true = torch.randn(3, 2, 32)
        metrics = SupervisedMICH._neural_recovery_metrics(true.clone(), true)
        assert metrics["val/neural/r2"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["val/neural/pearson"] == pytest.approx(1.0, abs=1e-4)
        assert metrics["val/neural/lag_samples"] == pytest.approx(0.0, abs=1e-6)

    def test_keys_present(self):
        true = torch.randn(2, 16)
        pred = torch.randn(2, 16)
        metrics = SupervisedMICH._neural_recovery_metrics(pred, true)
        assert set(metrics.keys()) == {
            "val/neural/r2",
            "val/neural/pearson",
            "val/neural/lag_samples",
        }
        assert all(isinstance(v, float) for v in metrics.values())

    @pytest.mark.parametrize("k", [1, 3, 5, -3])
    def test_pure_delay_detected_in_lag(self, k):
        """pred == true delayed by k samples (pred[i] = true[i-k] via torch.roll).

        Empirically, this module's FFT cross-correlation (true * conj(pred))
        convention yields peak_lag == -k for this construction (verified by
        direct calculation prior to writing this test). White noise gives a
        clean, unambiguous xcorr peak.
        """
        T = 32
        true = torch.randn(1, T)
        pred = torch.roll(true, shifts=k, dims=-1)
        metrics = SupervisedMICH._neural_recovery_metrics(pred, true)
        assert metrics["val/neural/lag_samples"] == pytest.approx(-k, abs=1e-6)
        # A pure relabeling (roll) still gives perfect r2/pearson once the
        # correct lag is discounted -- but r2/pearson here are computed
        # WITHOUT lag correction, so they need not be 1 for k != 0. Just
        # check they're finite.
        assert torch.isfinite(torch.tensor(metrics["val/neural/r2"]))
        assert torch.isfinite(torch.tensor(metrics["val/neural/pearson"]))

    def test_shape_with_multiple_leading_dims_flattened(self):
        """Leading dims (e.g. [N, L]) are flattened to rows of length T before reduction."""
        pred = torch.randn(5, 3, 20)
        true = torch.randn(5, 3, 20)
        metrics = SupervisedMICH._neural_recovery_metrics(pred, true)
        # Just confirm it runs on multi-dim leading shape and returns finite scalars.
        assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


# -------------------------
# forward()
# -------------------------


class TestForward:
    def test_normaliser_none_passes_bold_through_unchanged(self):
        captured = {}

        class _RecordingNet(torch.nn.Module):
            def forward(self, x):
                captured["input"] = x
                return x

        model = _make_model(net=_RecordingNet(), normaliser=None)
        bold = torch.randn(_B, _L, _T, _H, _W)
        out = model(bold)
        assert torch.equal(captured["input"], bold)
        assert torch.equal(out, bold)

    def test_normaliser_invoked_and_its_output_fed_to_net(self):
        captured = {}

        class _RecordingNet(torch.nn.Module):
            def forward(self, x):
                captured["input"] = x
                return x

        class _AddNormaliser:
            """Recognizable transform: adds a fixed offset."""

            def __init__(self, offset: float):
                self.offset = offset
                self.calls = []

            def normalize(self, x):
                self.calls.append(x)
                return x + self.offset

        normaliser = _AddNormaliser(offset=100.0)
        model = _make_model(net=_RecordingNet(), normaliser=normaliser)
        bold = torch.randn(_B, _L, _T, _H, _W)
        out = model(bold)

        assert len(normaliser.calls) == 1
        assert torch.equal(normaliser.calls[0], bold)
        expected = bold + 100.0
        assert torch.equal(captured["input"], expected)
        assert torch.equal(out, expected)


# -------------------------
# training_step / validation_step thin-wrapper delegation
# -------------------------


class TestStepDelegation:
    def test_training_step_returns_shared_step_result_with_train_stage(self):
        model = _make_model()
        sentinel = torch.tensor(3.14)
        model._shared_step = MagicMock(return_value=sentinel)
        batch = _make_batch()

        result = model.training_step(batch, 0)

        model._shared_step.assert_called_once_with(batch, stage="train")
        assert result is sentinel

    def test_validation_step_calls_shared_step_with_val_stage_and_returns_none(self):
        model = _make_model()
        sentinel = torch.tensor(9.9)
        model._shared_step = MagicMock(return_value=sentinel)
        batch = _make_batch()

        result = model.validation_step(batch, 0)

        model._shared_step.assert_called_once_with(batch, stage="val")
        # validation_step discards _shared_step's return value.
        assert result is None


# -------------------------
# _shared_step (real execution, via Trainer) + buffer behaviour
# -------------------------


class _BufferRecorder(Callback):
    """Records SupervisedMICH's internal buffer lengths/shapes and train losses."""

    def __init__(self):
        self.train_buffer_lengths: list[int] = []
        self.val_buffer_lengths: list[int] = []
        self.train_losses: list[torch.Tensor] = []
        self.last_val_shapes: dict | None = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_buffer_lengths.append(len(pl_module._pred_buffer))
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.train_losses.append(loss)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.val_buffer_lengths.append(len(pl_module._pred_buffer))
        self.last_val_shapes = {
            "pred": pl_module._pred_buffer[-1].shape,
            "neural": pl_module._neural_buffer[-1].shape,
            "bold": pl_module._bold_buffer[-1].shape,
            "src_pos": pl_module._src_pos_buffer[-1].shape,
        }


@pytest.mark.slow
def test_shared_step_val_buffers_grow_train_buffers_untouched():
    """Two train batches leave buffers empty; two val batches grow them 1, then 2."""
    model = _make_model()
    train_batches = [_make_batch(), _make_batch()]
    val_batches = [_make_batch(), _make_batch()]

    recorder = _BufferRecorder()
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_train_batches=2,
        limit_val_batches=2,
        callbacks=[recorder],
    )
    trainer.fit(
        model,
        train_dataloaders=_make_loader(train_batches),
        val_dataloaders=_make_loader(val_batches),
    )

    assert recorder.train_buffer_lengths == [0, 0], "train stage must not touch val buffers"
    assert recorder.val_buffer_lengths == [1, 2], "val stage should append once per val batch"
    assert all(torch.isfinite(loss) for loss in recorder.train_losses)

    shapes = recorder.last_val_shapes
    assert shapes["pred"] == (_B, _L, _T, _H, _W)
    assert shapes["neural"] == (_B, _L, _T, _H, _W)
    assert shapes["bold"] == (_B, _L, _T, _H, _W)
    assert shapes["src_pos"] == (_B, 2)

    # Buffers are cleared by on_validation_epoch_end after the val epoch completes.
    assert model._pred_buffer == []
    assert model._neural_buffer == []
    assert model._bold_buffer == []
    assert model._src_pos_buffer == []


# -------------------------
# on_validation_epoch_end
# -------------------------


class TestOnValidationEpochEnd:
    def test_empty_buffer_returns_cleanly_without_trainer(self):
        """No prior validation_step calls => early return; no Trainer needed at all
        since self.log/self.trainer are never touched on this path."""
        model = _make_model()
        assert model._pred_buffer == []
        model.on_validation_epoch_end()  # should not raise
        assert model._pred_buffer == []

    @pytest.mark.slow
    def test_wandb_run_none_runs_fully_without_touching_run(self, monkeypatch):
        """Default state (no wandb.init() anywhere in this suite): metrics get
        computed/logged, but the image-logging branch must no-op cleanly."""
        monkeypatch.setattr(wandb, "run", None)
        model = _make_model()
        val_batches = [_make_batch(), _make_batch()]

        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        trainer.validate(model, dataloaders=_make_loader(val_batches))

        metrics = trainer.callback_metrics
        assert "val/neural/r2" in metrics
        assert "val/neural/pearson" in metrics
        assert "val/neural/lag_samples" in metrics
        for v in metrics.values():
            assert torch.isfinite(v)

        # Buffers cleared after epoch end.
        assert model._pred_buffer == []

    @pytest.mark.slow
    def test_wandb_run_mocked_logs_scalars_and_images_then_clears_buffers(self, monkeypatch):
        """With a fake wandb.run, run.log() must be called at least twice: once
        with the scalar recovery metrics, once with 'val/predictions' images.
        A second call with no new validation_step data must early-return
        cleanly (same as the empty-buffer path)."""
        fake_run = MagicMock()
        monkeypatch.setattr(wandb, "run", fake_run)

        model = _make_model()
        val_batches = [_make_batch(), _make_batch()]

        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        trainer.validate(model, dataloaders=_make_loader(val_batches))

        assert fake_run.log.call_count >= 2
        logged_dicts = [c.args[0] for c in fake_run.log.call_args_list]
        scalar_calls = [d for d in logged_dicts if "val/neural/r2" in d]
        image_calls = [d for d in logged_dicts if "val/predictions" in d]
        assert len(scalar_calls) == 1
        assert len(image_calls) == 1
        assert scalar_calls[0]["val/neural/pearson"] is not None
        assert scalar_calls[0]["val/neural/lag_samples"] is not None
        images = image_calls[0]["val/predictions"]
        assert len(images) == min(10, 2 * _B)  # subset = min(10, N)

        # Buffers cleared after the (single) validation epoch.
        assert model._pred_buffer == []

        # Second call with empty buffers: must not crash, matches the
        # empty-buffer no-op path (no new run.log calls).
        call_count_before = fake_run.log.call_count
        model.on_validation_epoch_end()
        assert fake_run.log.call_count == call_count_before


class TestOnValidationEpochEndSourcePairing:
    """Empirically pins down whether bold_src / true_src / pred_src plotting
    slices in on_validation_epoch_end are correctly paired per-sample.

    The task brief flagged a suspected indexing bug: `src_h`/`src_w` are
    computed once against the full concatenated buffer (length N) and then,
    for the plotting subset, re-used as `src_h[idx]` / `src_w[idx]` alongside
    `bold[idx, ...]`. Reading the source carefully (supervised.py lines
    161-179), `src_h`/`src_w` themselves ARE re-indexed by `idx` at the point
    they're used against `bold[idx]` (line 179: `bold[idx, :, :, src_h[idx],
    src_w[idx]]`), so the pairing looks correct on paper. This test verifies
    that empirically rather than trusting the read: each sample's bold and
    true-neural values are encoded so that they only agree if the plotting
    code extracted them at the SAME (sample, h, w) triple.
    """

    def test_bold_and_true_plot_slices_reference_the_same_sample_and_position(self, monkeypatch):
        N, H, W, L, T = 4, 4, 4, 1, 2
        OFFSET = 100_000.0

        def _identity(i, h, w):
            return float(i * 100 + h * 10 + w)

        bold = torch.zeros(N, L, T, H, W)
        neural = torch.zeros(N, L, T, H, W)
        # Distinct (h, w) source position per sample so a mismatched pairing
        # would decode to an inconsistent (i, h, w) triple.
        src_pos = torch.tensor([[i, (H - 1 - i)] for i in range(N)], dtype=torch.long)
        for i in range(N):
            for h in range(H):
                for w in range(W):
                    ident = _identity(i, h, w)
                    bold[i, :, :, h, w] = ident + OFFSET
                    neural[i, :, :, h, w] = ident

        model = _make_model(net=_mk_supervised_net(L=L))
        batch1 = {"bold": bold[:2], "neural": neural[:2], "source_position": src_pos[:2]}
        batch2 = {"bold": bold[2:], "neural": neural[2:], "source_position": src_pos[2:]}

        captured_calls: list[dict] = []

        def _fake_plot(**kwargs):
            captured_calls.append(kwargs)
            import matplotlib.pyplot as plt

            return plt.figure()

        monkeypatch.setattr("mich.models.supervised.plot_neural_bold_layers", _fake_plot)
        monkeypatch.setattr(wandb, "run", MagicMock())

        trainer = Trainer(
            max_epochs=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        trainer.validate(model, dataloaders=_make_loader([batch1, batch2]))

        # subset = min(10, N) = N here, so every sample is plotted exactly once.
        assert len(captured_calls) == N

        for kwargs in captured_calls:
            bold_val = kwargs["pred_bold"]
            true_val = kwargs["true_neural"]
            # pred_bold and true_bold are literally the same tensor in the
            # source (this baseline never predicts BOLD) -- confirm that too.
            assert torch.equal(kwargs["pred_bold"], kwargs["true_bold"])

            decoded_bold_ident = (bold_val - OFFSET).flatten()
            decoded_true_ident = true_val.flatten()
            # If bold/true were pulled from the same (sample, h, w) triple,
            # these must match exactly (both encode the same identity()).
            assert torch.allclose(decoded_bold_ident, decoded_true_ident, atol=1e-3), (
                "bold and true-neural plotting slices disagree -- would indicate "
                "a source-position pairing bug in on_validation_epoch_end"
            )

            # Further decode (i, h, w) from the value and check it matches a
            # crafted source_position row, i.e. the *correct* source position
            # for *some* sample was used (not e.g. always sample 0's).
            val = int(round(decoded_true_ident[0].item()))
            i_dec, h_dec, w_dec = val // 100, (val // 10) % 10, val % 10
            assert (h_dec, w_dec) == (int(src_pos[i_dec, 0]), int(src_pos[i_dec, 1]))

        # No indexing bug found: pairing is self-consistent for every plotted
        # sample. (Reported back regardless, per task instructions.)


# -------------------------
# configure_optimizers
# -------------------------


class TestConfigureOptimizers:
    def test_returns_expected_structure(self):
        model = _make_model()
        out = model.configure_optimizers()

        assert set(out.keys()) == {"optimizer", "lr_scheduler"}
        assert isinstance(out["optimizer"], torch.optim.Adam)
        lr_sched = out["lr_scheduler"]
        assert isinstance(lr_sched["scheduler"], torch.optim.lr_scheduler.LambdaLR)
        assert lr_sched["scheduler"].optimizer is out["optimizer"]
        assert lr_sched["interval"] == "step"
        assert lr_sched["frequency"] == 1


# -------------------------
# Full pipeline: Trainer.fit
# -------------------------


@pytest.mark.slow
def test_training_fast_dev_run_completes_without_error():
    """One train step + one val step + val epoch end complete without error,
    using a bare DataLoader of dict batches (mirrors test_mich.py's fast_dev_run
    smoke test but with the simpler batch schema _shared_step needs)."""
    model = _make_model()
    train_loader = _make_loader([_make_batch()])
    val_loader = _make_loader([_make_batch()])

    trainer = Trainer(
        max_epochs=1,
        fast_dev_run=True,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    # Should complete without raising.
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


@pytest.mark.slow
def test_training_fast_dev_run_loss_finite():
    model = _make_model()
    train_loader = _make_loader([_make_batch()])
    val_loader = _make_loader([_make_batch()])

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
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    assert len(captured) == 1
    out = captured[0]
    if isinstance(out, torch.Tensor):
        loss_val = out.item()
    elif isinstance(out, dict) and "loss" in out:
        loss_val = out["loss"].item()
    else:
        loss_val = 0.0
    assert torch.isfinite(torch.tensor(loss_val))
