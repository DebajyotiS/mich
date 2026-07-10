from __future__ import annotations

import math
import types

import pytest
import torch

from mich.data.balloon import AcquisitionConstants
from mich.models.physio import LearnablePhysioMixin


class _Dummy(LearnablePhysioMixin, torch.nn.Module):
    def __init__(self, learnable_physio=None, **hparam_kwargs):
        super().__init__()
        self.hparams = types.SimpleNamespace(**hparam_kwargs)
        self._setup_learnable_physio(learnable_physio)


def _mk_hparams(**overrides):
    kwargs = dict(
        haemo=types.SimpleNamespace(kappa=0.65, gamma=0.41, alpha=0.32, tau=1.0),
        acquisition=types.SimpleNamespace(k1=0.02, k2=0.38, k3=0.38, E0=0.35),
        V0=0.02,
    )
    kwargs.update(overrides)
    return kwargs


def _mk_dummy(learnable_physio=None, **overrides):
    return _Dummy(learnable_physio, **_mk_hparams(**overrides))


# -----------------------------
# _setup_learnable_physio
# -----------------------------


@pytest.mark.parametrize("learnable_physio", [None, {}])
def test_setup_learnable_physio_none_or_empty_registers_nothing(learnable_physio):
    m = _mk_dummy(learnable_physio)
    assert list(m.parameters()) == []
    for name in ("kappa", "gamma", "alpha", "tau", "V0", "E0"):
        assert not hasattr(m, f"_physio_log_{name}")


def test_setup_learnable_physio_mixed_registers_only_flagged_names():
    m = _mk_dummy({"kappa": True, "tau": True, "gamma": False})

    assert isinstance(m._physio_log_kappa, torch.nn.Parameter)
    assert m._physio_log_kappa.requires_grad
    assert torch.isclose(torch.exp(m._physio_log_kappa), torch.tensor(0.65))

    assert isinstance(m._physio_log_tau, torch.nn.Parameter)
    assert m._physio_log_tau.requires_grad
    assert torch.isclose(torch.exp(m._physio_log_tau), torch.tensor(1.0))

    for name in ("gamma", "alpha", "V0", "E0"):
        assert not hasattr(m, f"_physio_log_{name}")

    names = {n for n, _ in m.named_parameters()}
    assert names == {"_physio_log_kappa", "_physio_log_tau"}


@pytest.mark.parametrize("bad_val", [0.0, -1.0])
def test_setup_learnable_physio_raises_for_non_positive_haemo_group_value(bad_val):
    with pytest.raises(ValueError, match=r"learnable_physio\.alpha: init value must be > 0"):
        _mk_dummy(
            {"alpha": True},
            haemo=types.SimpleNamespace(kappa=0.65, gamma=0.41, alpha=bad_val, tau=1.0),
        )


@pytest.mark.parametrize("bad_val", [0.0, -1.0])
def test_setup_learnable_physio_raises_for_non_positive_top_level_value(bad_val):
    with pytest.raises(ValueError, match=r"learnable_physio\.V0: init value must be > 0"):
        _mk_dummy({"V0": True}, V0=bad_val)


# -----------------------------
# _physio
# -----------------------------


def test_physio_returns_fixed_hparam_unchanged_when_not_learnable():
    m = _mk_dummy(None)
    assert m._physio("kappa") == 0.65
    assert m._physio("V0") == 0.02
    assert not isinstance(m._physio("kappa"), torch.Tensor)


def test_physio_returns_exp_of_learnable_param_matching_init_value():
    m = _mk_dummy({"kappa": True})
    val = m._physio("kappa")
    assert isinstance(val, torch.Tensor)
    assert torch.isclose(val, torch.tensor(0.65))


def test_physio_learnable_value_is_differentiable_wrt_log_param():
    m = _mk_dummy({"kappa": True})
    val = m._physio("kappa")
    assert val.requires_grad
    (val.sum() * 2.0).backward()
    assert m._physio_log_kappa.grad is not None


# -----------------------------
# _current_acquisition
# -----------------------------


def test_current_acquisition_fixed_e0_returns_hparams_verbatim_without_recomputing():
    # deliberately inconsistent with the k1/k2/k3-from-E0 formula, to prove the
    # fixed branch doesn't silently recompute.
    m = _mk_dummy(
        None,
        acquisition=types.SimpleNamespace(k1=111.0, k2=222.0, k3=333.0, E0=0.35),
    )
    ac = m._current_acquisition()
    assert ac == AcquisitionConstants(k1=111.0, k2=222.0, k3=333.0)


def test_current_acquisition_learnable_e0_recomputes_k1_k2_k3_from_formula():
    f0, TE, eps, r0, E0 = 0.5, 0.028, 0.4, 25.0, 0.35
    m = _mk_dummy(
        {"E0": True},
        acquisition=types.SimpleNamespace(
            k1=0.0, k2=0.0, k3=0.0, E0=E0, f0=f0, TE=TE, eps=eps, r0=r0
        ),
    )
    ac = m._current_acquisition()

    expected_k1 = 4.3 * f0 * E0 * TE
    expected_k2 = eps * r0 * E0 * TE
    expected_k3 = 1.0 - eps

    assert math.isclose(float(ac.k1.detach()), expected_k1, rel_tol=1e-6)
    assert math.isclose(float(ac.k2.detach()), expected_k2, rel_tol=1e-6)
    assert math.isclose(float(ac.k3), expected_k3, rel_tol=1e-6)


def test_current_acquisition_learnable_e0_uses_current_not_init_value():
    f0, TE, eps, r0 = 0.5, 0.028, 0.4, 25.0
    m = _mk_dummy(
        {"E0": True},
        acquisition=types.SimpleNamespace(
            k1=0.0, k2=0.0, k3=0.0, E0=0.35, f0=f0, TE=TE, eps=eps, r0=r0
        ),
    )
    with torch.no_grad():
        m._physio_log_E0.add_(0.1)  # move the parameter away from its init value

    new_E0 = torch.exp(m._physio_log_E0).item()
    ac = m._current_acquisition()
    assert math.isclose(float(ac.k1.detach()), 4.3 * f0 * new_E0 * TE, rel_tol=1e-6)
    assert math.isclose(float(ac.k2.detach()), eps * r0 * new_E0 * TE, rel_tol=1e-6)


def test_current_acquisition_learnable_e0_is_differentiable_wrt_log_e0():
    m = _mk_dummy(
        {"E0": True},
        acquisition=types.SimpleNamespace(
            k1=0.0, k2=0.0, k3=0.0, E0=0.35, f0=0.5, TE=0.028, eps=0.4, r0=25.0
        ),
    )
    ac = m._current_acquisition()
    (ac.k1 + ac.k2 + ac.k3).backward()
    assert m._physio_log_E0.grad is not None
