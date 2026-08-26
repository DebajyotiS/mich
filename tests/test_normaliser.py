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
        # h != w in every position: symmetric positions make an h/w swap in
        # _gather_neighbourhood's offset arithmetic undetectable.
        pos = torch.tensor([[[5, 2]], [[3, 6]], [[7, 1]]], dtype=torch.long)  # [B, S=1, 2]
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

    def test_neighbourhood_gathers_the_asymmetric_source_position(self):
        """The gather must read (h, w) in that order, on a non-square grid.

        The test above plants its value at (4, 4) on an 8x8 grid, so swapping the
        h and w lookups reads the very same voxel and cannot be detected. Here the
        grid is 6x9 and the position is (1, 7), which is only in range when h and
        w are used the right way round.
        """
        B, L, T, H, W, r = 1, 1, 1, 6, 9, 1
        norm = _make_norm(H=H, W=W, radius=r)
        bold = torch.zeros(B, L, T, H, W)
        src_h, src_w = 1, 7
        bold[0, 0, 0, src_h, src_w] = 99.0
        # A decoy at the transposed location would be picked up by a swapped gather.
        # (7, 1) is out of range for h, so plant the decoy where a swap would clamp to.
        bold[0, 0, 0, H - 1, 1] = -55.0
        pos = torch.tensor([[[src_h, src_w]]], dtype=torch.long)
        mask = torch.ones(B, 1, dtype=torch.bool)

        out = norm._gather_neighbourhood(bold, pos, mask)
        assert (out == 99.0).any(), "the true source voxel must be gathered"
        assert not (out == -55.0).any(), "a transposed gather must not reach the decoy"


# -------------------------
# Phase 2 oracles: which statistics are actually used, and the shared-scale contract
#
# The tests above verify that the running statistics move and that outputs stay
# finite. They never check *which* statistics a given call normalises with, which
# is the module's central documented semantic, nor the single-shared-scale design
# rationale. Both are pinned below.
# -------------------------


class TestWhichStatisticsAreUsed:
    def test_training_normalises_with_this_batch_statistics_not_running(self):
        """While training (and not frozen or paused), `forward` must normalise with
        this batch's own mean/std, per the docstring, and not the running values.

        Written as a differential check to avoid restating the source's own
        gather-and-average logic: two normalisers that differ *only* in their
        running statistics must produce identical output for the same batch, because
        the running values are not supposed to enter the computation at all. The
        running counts are large so that blending this batch in cannot quietly drag
        the two running means together and mask the difference.
        """

        def _norm_with_running(mean_val):
            n = _make_norm(H=8, W=8, radius=2)
            n.train()
            n.running_mean.fill_(mean_val)
            n.running_M2.fill_(1e6 - 1.0)  # var = M2/(count-1) = 1.0
            n.running_count.fill_(1e6)
            return n

        norm_a = _norm_with_running(0.0)
        norm_b = _norm_with_running(100.0)

        torch.manual_seed(0)
        bold = torch.randn(2, 3, 4, 8, 8) + 7.0
        pos = torch.tensor([[[4, 3]], [[2, 5]]], dtype=torch.long)
        num_sources = torch.ones(2, dtype=torch.long)

        out_a = norm_a(bold.clone(), pos, num_sources)
        out_b = norm_b(bold.clone(), pos, num_sources)

        assert torch.allclose(out_a, out_b, atol=1e-6), (
            "training output must not depend on the running statistics"
        )
        # Guard against the comparison being vacuous because both saturated the clamp.
        assert out_a.abs().max() < 10.0

    def test_eval_normalises_with_running_statistics_and_does_not_update(self):
        """In eval mode the running statistics must be used verbatim and left
        untouched, so the result equals `normalize` exactly."""
        norm = _make_norm(H=8, W=8, radius=2)
        norm.running_mean.fill_(2.0)
        norm.running_M2.fill_(4.0)
        norm.running_count.fill_(5)  # var = 1.0
        norm.eval()

        bold = torch.linspace(0.0, 4.0, 2 * 3 * 4 * 8 * 8).reshape(2, 3, 4, 8, 8)
        before = (
            norm.running_mean.clone(),
            norm.running_M2.clone(),
            norm.running_count.clone(),
            norm.step.clone(),
        )

        out = norm(bold)

        assert torch.allclose(out, norm.normalize(bold))
        assert torch.equal(norm.running_mean, before[0])
        assert torch.equal(norm.running_M2, before[1])
        assert torch.equal(norm.running_count, before[2])
        assert torch.equal(norm.step, before[3]), "eval must not advance the step counter"

    def test_pause_update_uses_running_statistics_while_still_training(self):
        """`pause_update=True` must switch to the running statistics, matching
        `normalize`, not merely skip the counter increment."""
        norm = _make_norm(H=8, W=8, radius=2)
        norm.train()
        norm.running_mean.fill_(2.0)
        norm.running_M2.fill_(4.0)
        norm.running_count.fill_(5)

        bold = torch.linspace(0.0, 4.0, 2 * 3 * 4 * 8 * 8).reshape(2, 3, 4, 8, 8)
        pos = torch.tensor([[[4, 3]], [[2, 5]]], dtype=torch.long)
        num_sources = torch.ones(2, dtype=torch.long)

        out = norm(bold, pos, num_sources, pause_update=True)
        assert torch.allclose(out, norm.normalize(bold))
        assert norm.step.item() == 0


class TestSharedScaleContract:
    def test_all_layers_share_one_scalar_scale(self):
        """The module docstring's central design claim: a single shared scalar is
        applied across sources, layers, time and voxels, so that inter-layer
        amplitude ratios survive normalisation.

        Give two layers deliberately different amplitudes and check the ratio
        between them is preserved exactly. A per-layer scale would equalise them.
        """
        norm = _make_norm(H=8, W=8, radius=2)
        norm.running_mean.fill_(0.0)
        norm.running_M2.fill_(16.0)
        norm.running_count.fill_(5)  # var = 4, std = 2 (non-unit on purpose)
        norm.eval()

        bold = torch.zeros(1, 3, 2, 8, 8)
        bold[:, 0] = 1.0
        bold[:, 1] = 3.0
        bold[:, 2] = 5.0

        out = norm(bold)
        # Ratios between layers must be unchanged by a single shared scale.
        assert out[:, 1].mean() / out[:, 0].mean() == pytest.approx(3.0, rel=1e-6)
        assert out[:, 2].mean() / out[:, 0].mean() == pytest.approx(5.0, rel=1e-6)

    def test_normalize_denormalize_roundtrip_is_exact_and_unguarded(self):
        """An unconditional round-trip oracle.

        The existing round-trip test wraps its assertion in `if in_range.all():`,
        so it silently becomes a no-op if the values ever saturate the [-10, 10]
        clamp. Here the inputs are chosen to sit well inside the clamp and the
        assertion always runs.
        """
        norm = _make_norm()
        norm.running_mean.fill_(2.0)
        # var = M2 / (count - 1) = 16 / 4 = 4, so std = 2. A non-unit std is what
        # makes a `/ std` vs `* std` mix-up observable at all.
        norm.running_M2.fill_(16.0)
        norm.running_count.fill_(5)

        bold = torch.linspace(-1.0, 5.0, 32).reshape(1, 1, 2, 4, 4)
        normalized = norm.normalize(bold)
        # Guard the guard: prove we are inside the clamp rather than assuming it.
        assert normalized.abs().max() < 9.0
        assert torch.allclose(norm.denormalize(normalized), bold, atol=1e-5)

    def test_denormalize_inverts_the_exact_affine_transform(self):
        """denormalize(x) = x * std + mean with the running statistics, pinned
        against hand-computed values so a swapped mean/std or a dropped term is
        visible. With mean=3, M2=16, count=5: var = 16/4 = 4, std = 2.
        Hence denormalize(1.5) = 1.5 * 2 + 3 = 6.
        """
        norm = _make_norm()
        norm.running_mean.fill_(3.0)
        norm.running_M2.fill_(16.0)
        norm.running_count.fill_(5)

        out = norm.denormalize(torch.tensor([[[[[1.5]]]]]))
        assert out.item() == pytest.approx(6.0, rel=1e-6)

    def test_training_normalisation_is_invariant_to_rescaling_the_batch(self):
        """Normalising by (x - mean) / std is exactly scale-invariant: scaling the
        input by c scales both the batch mean and the batch std by c, so the
        output must be unchanged.

        This pins the `sqrt` on the batch variance without restating how the
        source computes it. Dropping the sqrt makes the divisor scale as c^2
        instead of c, so the output would shrink by 1/c.
        """

        def _fresh():
            n = _make_norm(H=8, W=8, radius=2)
            n.train()
            n.running_mean.fill_(0.0)
            n.running_M2.fill_(1e6 - 1.0)
            n.running_count.fill_(1e6)
            return n

        torch.manual_seed(0)
        bold = torch.randn(2, 3, 4, 8, 8) * 1.5 + 4.0
        pos = torch.tensor([[[4, 3]], [[2, 5]]], dtype=torch.long)
        num_sources = torch.ones(2, dtype=torch.long)

        c = 3.0
        out_plain = _fresh()(bold.clone(), pos, num_sources)
        out_scaled = _fresh()(bold.clone() * c, pos, num_sources)

        assert out_plain.abs().max() < 10.0, "clamp saturated; comparison would be vacuous"
        assert torch.allclose(out_plain, out_scaled, atol=1e-5), (
            "normalisation must be invariant to a global rescaling of the batch"
        )
