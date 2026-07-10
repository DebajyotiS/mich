"""Tests for src/models/normaliser.py LayerwiseBOLDNormalizer."""

from __future__ import annotations

import pytest
import torch
from mich.models.normaliser import LayerwiseBOLDNormalizer

# -------------------------
# Helpers
# -------------------------


def _make_norm(
    H: int = 8,
    W: int = 8,
    freeze_after: int = 5000,
    radius: int = 2,
    eps: float = 1e-6,
) -> LayerwiseBOLDNormalizer:
    return LayerwiseBOLDNormalizer(
        H=H, W=W, eps=eps, freeze_after_steps=freeze_after, neighbourhood_radius=radius
    )


def _bold_and_pos(
    B: int = 2, L: int = 3, T: int = 4, H: int = 8, W: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bold = torch.randn(B, L, T, H, W)
    pos = torch.tensor([[[H // 2, W // 2]]] * B, dtype=torch.long)  # [B, S=1, 2]
    num_sources = torch.ones(B, dtype=torch.long)
    return bold, pos, num_sources


# -------------------------
# frozen property
# -------------------------


class TestFrozenProperty:
    def test_not_frozen_at_init(self):
        norm = _make_norm(freeze_after=10)
        assert not norm.frozen

    def test_frozen_when_step_equals_threshold(self):
        norm = _make_norm(freeze_after=3)
        norm.step.fill_(3)
        assert norm.frozen

    def test_frozen_when_step_exceeds_threshold(self):
        norm = _make_norm(freeze_after=3)
        norm.step.fill_(100)
        assert norm.frozen

    def test_not_frozen_one_step_before_threshold(self):
        norm = _make_norm(freeze_after=3)
        norm.step.fill_(2)
        assert not norm.frozen


# -------------------------
# running_var property
# -------------------------


class TestRunningVar:
    def test_running_var_returns_ones_when_count_is_zero(self):
        norm = _make_norm()
        assert norm.running_count.item() == 0
        var = norm.running_var
        assert torch.allclose(var, torch.ones_like(var))

    def test_running_var_returns_ones_when_count_is_one(self):
        norm = _make_norm()
        norm.running_count.fill_(1)
        norm.running_M2.fill_(99.0)  # irrelevant for count < 2
        var = norm.running_var
        assert torch.allclose(var, torch.ones_like(var))

    def test_running_var_computed_correctly_when_count_ge_2(self):
        norm = _make_norm()
        norm.running_count.fill_(5)
        norm.running_M2.fill_(8.0)
        # var = M2 / (count - 1) = 8 / 4 = 2.0
        var = norm.running_var
        assert torch.allclose(var, torch.tensor(2.0), atol=1e-6)


# -------------------------
# forward — eval mode
# -------------------------


class TestForwardEval:
    def test_eval_uses_running_stats_not_batch(self):
        norm = _make_norm()
        norm.eval()
        step_before = norm.step.item()
        bold = torch.randn(2, 3, 4, 8, 8)
        _ = norm(bold)
        # step not incremented in eval
        assert norm.step.item() == step_before

    def test_eval_at_init_mean0_std1_identity_clamp(self):
        norm = _make_norm()
        norm.eval()
        bold = torch.zeros(1, 1, 1, 8, 8) + 5.0
        out = norm(bold)
        # mean=0, std=1 -> (bold - 0) / 1 = bold; within clamp range
        assert torch.allclose(out, bold, atol=1e-5)

    def test_eval_output_clamped_at_plus_minus_10(self):
        norm = _make_norm()
        norm.eval()
        bold = torch.full((1, 1, 1, 8, 8), 100.0)
        out = norm(bold)
        assert out.max().item() <= 10.0

        bold_neg = torch.full((1, 1, 1, 8, 8), -100.0)
        out_neg = norm(bold_neg)
        assert out_neg.min().item() >= -10.0

    def test_eval_output_shape_matches_input(self):
        norm = _make_norm()
        norm.eval()
        bold = torch.randn(3, 4, 5, 8, 8)
        out = norm(bold)
        assert out.shape == bold.shape

    def test_eval_preserves_float32_dtype(self):
        norm = _make_norm()
        norm.eval()
        bold = torch.randn(2, 3, 4, 8, 8, dtype=torch.float32)
        out = norm(bold)
        assert out.dtype == torch.float32

    def test_eval_preserves_float16_dtype(self):
        norm = _make_norm()
        norm.eval()
        bold = torch.randn(2, 3, 4, 8, 8, dtype=torch.float16)
        out = norm(bold)
        assert out.dtype == torch.float16


# -------------------------
# forward — training mode
# -------------------------


class TestForwardTrain:
    def test_training_without_source_position_raises(self):
        norm = _make_norm()
        norm.train()
        bold = torch.randn(2, 3, 4, 8, 8)
        with pytest.raises(ValueError, match="source_position and num_sources are required"):
            norm(bold, source_position=None, num_sources=None)

    def test_training_without_num_sources_raises(self):
        norm = _make_norm()
        norm.train()
        bold, pos, _ = _bold_and_pos()
        with pytest.raises(ValueError, match="source_position and num_sources are required"):
            norm(bold, source_position=pos, num_sources=None)

    def test_training_increments_step(self):
        norm = _make_norm()
        norm.train()
        bold, pos, num_sources = _bold_and_pos()
        norm(bold, source_position=pos, num_sources=num_sources)
        assert norm.step.item() == 1

    def test_training_increments_running_count(self):
        norm = _make_norm()
        norm.train()
        bold, pos, num_sources = _bold_and_pos()
        norm(bold, source_position=pos, num_sources=num_sources)
        assert norm.running_count.item() > 0

    def test_training_output_finite(self):
        norm = _make_norm()
        norm.train()
        bold, pos, num_sources = _bold_and_pos()
        out = norm(bold, source_position=pos, num_sources=num_sources)
        assert torch.isfinite(out).all()

    def test_pause_update_skips_welford_and_step(self):
        norm = _make_norm()
        norm.train()
        bold, pos, num_sources = _bold_and_pos()
        norm(bold, source_position=pos, num_sources=num_sources, pause_update=True)
        assert norm.step.item() == 0
        assert norm.running_count.item() == 0

    def test_frozen_training_skips_update(self):
        norm = _make_norm(freeze_after=0)
        norm.train()
        bold, pos, num_sources = _bold_and_pos()
        norm(bold, source_position=pos, num_sources=num_sources)
        # step starts at 0, freeze_after=0 -> already frozen -> no update
        assert norm.step.item() == 0
        assert norm.running_count.item() == 0

    def test_running_mean_converges_toward_data_mean(self):
        """After many batches of constant-valued signal, mean should approach that constant."""
        norm = _make_norm(freeze_after=1000)
        norm.train()
        target = 3.0
        bold = torch.full((4, 2, 4, 8, 8), target)
        pos = torch.tensor([[[4, 4]]] * 4, dtype=torch.long)
        num_sources = torch.ones(4, dtype=torch.long)
        for _ in range(20):
            norm(bold, source_position=pos, num_sources=num_sources)
        assert abs(norm.running_mean.item() - target) < 0.5


# -------------------------
# normalize / denormalize
# -------------------------


class TestNormalizeAndDenormalize:
    def test_normalize_output_is_finite(self):
        norm = _make_norm()
        bold = torch.randn(2, 3, 4, 8, 8)
        out = norm.normalize(bold)
        assert torch.isfinite(out).all()

    def test_denormalize_output_is_finite(self):
        norm = _make_norm()
        bold_norm = torch.randn(2, 3, 4, 8, 8)
        out = norm.denormalize(bold_norm)
        assert torch.isfinite(out).all()

    def test_normalize_denormalize_roundtrip_for_moderate_values(self):
        """Values that don't hit the [-10, 10] clamp should survive the roundtrip."""
        norm = _make_norm()
        # Give non-trivial running stats
        norm.running_mean.fill_(2.0)
        norm.running_M2.fill_(4.0)
        norm.running_count.fill_(5)  # var = 4/4 = 1.0, std = 1.0

        # Construct bold values whose normalized form stays within clamp
        bold = torch.linspace(0.0, 4.0, 16).reshape(1, 1, 1, 4, 4)
        normalized = norm.normalize(bold)
        reconstructed = norm.denormalize(normalized)

        in_range = (normalized > -10.0) & (normalized < 10.0)
        if in_range.all():
            assert torch.allclose(reconstructed.float(), bold.float(), atol=1e-4)

    def test_normalize_output_shape(self):
        norm = _make_norm()
        bold = torch.randn(2, 3, 5, 8, 8)
        assert norm.normalize(bold).shape == bold.shape

    def test_denormalize_output_shape(self):
        norm = _make_norm()
        bold_norm = torch.randn(2, 3, 5, 8, 8)
        assert norm.denormalize(bold_norm).shape == bold_norm.shape


# -------------------------
# _gather_neighbourhood
# -------------------------


class TestGatherNeighbourhood:
    def test_output_shape_is_m_l_t_n(self):
        B, L, T, H, W, r = 3, 4, 5, 10, 10, 2
        norm = _make_norm(H=H, W=W, radius=r)
        bold = torch.randn(B, L, T, H, W)
        pos = torch.tensor([[[5, 5]], [[3, 3]], [[7, 7]]], dtype=torch.long)  # [B, S=1, 2]
        mask = torch.ones(B, 1, dtype=torch.bool)
        out = norm._gather_neighbourhood(bold, pos, mask)
        N = (2 * r + 1) ** 2
        assert out.shape == (B, L, T, N)

    def test_masked_out_sources_are_excluded(self):
        B, S, L, T, H, W, r = 2, 3, 1, 1, 8, 8, 1
        norm = _make_norm(H=H, W=W, radius=r)
        bold = torch.randn(B, L, T, H, W)
        pos = torch.randint(0, H, (B, S, 2), dtype=torch.long)
        num_sources = torch.tensor([1, 3])
        mask = norm._source_mask(num_sources, S)
        out = norm._gather_neighbourhood(bold, pos, mask)
        assert out.shape[0] == int(mask.sum().item())

    def test_out_of_bounds_position_does_not_raise(self):
        B, L, T, H, W, r = 1, 1, 1, 8, 8, 3
        norm = _make_norm(H=H, W=W, radius=r)
        bold = torch.randn(B, L, T, H, W)
        pos = torch.tensor([[[0, 0]]], dtype=torch.long)  # corner; offsets go negative -> clamped
        mask = torch.ones(B, 1, dtype=torch.bool)
        out = norm._gather_neighbourhood(bold, pos, mask)
        assert torch.isfinite(out).all()

    def test_neighbourhood_at_center_contains_known_value(self):
        """Exact center voxel should always appear in the neighbourhood."""
        B, L, T, H, W, r = 1, 1, 1, 8, 8, 1
        norm = _make_norm(H=H, W=W, radius=r)
        bold = torch.zeros(B, L, T, H, W)
        center_h, center_w = 4, 4
        bold[0, 0, 0, center_h, center_w] = 99.0
        pos = torch.tensor([[[center_h, center_w]]], dtype=torch.long)
        mask = torch.ones(B, 1, dtype=torch.bool)
        out = norm._gather_neighbourhood(bold, pos, mask)
        assert (out == 99.0).any()
