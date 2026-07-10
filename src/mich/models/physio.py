"""Optionally-learnable physiological constants (kappa/gamma/alpha/tau/V0/E0).

Parameterized in log-space so gradient descent can't push a physically positive-only
quantity negative; `_physio()` exponentiates back on read. Any name not flagged True in
`learnable_physio` stays a fixed hparam.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch

from mich.data.balloon import AcquisitionConstants


class LearnablePhysioMixin:
    # name -> (hparams group, key) for the fixed value used as init / fallback.
    # group is None for top-level hparams (e.g. V0).
    _PHYSIO_HPARAM_PATH: dict[str, tuple[str | None, str]] = {
        "kappa": ("haemo", "kappa"),
        "gamma": ("haemo", "gamma"),
        "alpha": ("haemo", "alpha"),
        "tau": ("haemo", "tau"),
        "V0": (None, "V0"),
        "E0": ("acquisition", "E0"),
    }

    def _setup_learnable_physio(self, learnable_physio: Mapping[str, bool] | None) -> None:
        learnable_physio = dict(learnable_physio or {})
        for name, (group, key) in self._PHYSIO_HPARAM_PATH.items():
            if not learnable_physio.get(name, False):
                continue
            src = getattr(self.hparams, group) if group else self.hparams
            init_val = float(getattr(src, key))
            if init_val <= 0:
                raise ValueError(f"learnable_physio.{name}: init value must be > 0, got {init_val}")
            self.register_parameter(
                f"_physio_log_{name}", torch.nn.Parameter(torch.tensor(math.log(init_val)))
            )

    def _physio(self, name: str) -> torch.Tensor | float:
        """Current value of a physio constant -- the learnable log-space parameter if
        `learnable_physio.<name>` was set True at construction, else the fixed hparam."""
        log_param = getattr(self, f"_physio_log_{name}", None)
        if log_param is not None:
            return torch.exp(log_param)
        group, key = self._PHYSIO_HPARAM_PATH[name]
        src = getattr(self.hparams, group) if group else self.hparams
        return getattr(src, key)

    def _current_acquisition(self) -> AcquisitionConstants:
        """k1/k2/k3, recomputed from a learnable E0 if applicable (otherwise the fixed,
        precomputed values injected from the simulation config)."""
        ac = self.hparams.acquisition
        if getattr(self, "_physio_log_E0", None) is None:
            return AcquisitionConstants(k1=ac.k1, k2=ac.k2, k3=ac.k3)
        E0 = self._physio("E0")
        k1 = 4.3 * ac.f0 * E0 * ac.TE
        k2 = ac.eps * ac.r0 * E0 * ac.TE
        k3 = 1.0 - ac.eps
        return AcquisitionConstants(k1=k1, k2=k2, k3=k3)
