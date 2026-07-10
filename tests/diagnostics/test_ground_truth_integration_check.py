"""Apparatus-independent re-check of the x_true<->s_true/f_true alignment claim.

Every prior "ground-truth closure" check (the original shift search, P4b, LS7) routed
through `dt_physical` (central-difference FD) applied to s_true, and LS7 showed that
operator alone produces a ~0.75-sample apparent lag against a quantity that is analytically
equal to it (`balloon_derivatives`'s ds_dt is literally `x - kappa*s - gamma*(f-1)`, the ODE
RHS evaluated on whatever x,s,f you hand it -- not an independent derivative; P4b's rel_gap
and LS7's lag are the same residual in different units). So none of those checks actually
confirm zero true offset between x_true and s_true/f_true independent of the FD apparatus.

This file instead replicates the generator's own RK4 sub-step INTEGRATION (the same method
LS3 uses for bold/v,q, extended to s,f, which simulate_cortex's output already contains) --
no differentiation anywhere. If the re-simulated s,f trajectory (driven purely by x_true,
starting from the same zero initial condition the generator uses) matches recorded s_true/
f_true tightly with zero shift, and every non-zero integer shift of the x_true input makes
it worse, that's an apparatus-free confirmation. If it doesn't, the earlier "no timestamp
offset" conclusion needs to be retracted, not just caveated.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from common import rms, write_report
from mich.data.balloon import CortexLayer, HaemodynamicConstants, HaemodynamicState, simulate_cortex

SHIFT_RANGE = range(-3, 4)


def test_resimulated_s_f_match_recorded_at_zero_shift(diag_model, diag_out_dir):
    model, full_cfg = diag_model

    h5_path = full_cfg.datamodule.data.path
    with h5py.File(str(h5_path), "r") as f:
        sim_cfg = json.loads(f["meta"].attrs["config"])["simulation"]
        neural = torch.from_numpy(f["layer_0"]["x"][:200]).float()  # [N, T, H, W]
        s_rec = torch.from_numpy(f["layer_0"]["s"][:200]).float()
        f_rec = torch.from_numpy(f["layer_0"]["f"][:200]).float()
        src_pos = torch.from_numpy(f["meta"]["sources"]["position"][:200]).long()  # [N, 1, 2]

    dt_sim, haemo_dt = float(sim_cfg["dt"]), float(sim_cfg["haemo_dt"])
    haemo_ratio = round(dt_sim / haemo_dt)

    haemo = full_cfg.model.haemo
    constants = HaemodynamicConstants(
        kappa=haemo.kappa,
        gamma=haemo.gamma,
        alpha=haemo.alpha,
        E0=full_cfg.model.acquisition.E0,
        V0=full_cfg.model.V0,
    )

    N = neural.shape[0]
    report_per_shift = {k: [] for k in SHIFT_RANGE}
    for i in range(N):
        h, w = int(src_pos[i, 0, 0]), int(src_pos[i, 0, 1])
        x_row = neural[i, :, h, w].numpy().astype(np.float64)
        s_true_row = s_rec[i, :, h, w].numpy().astype(np.float64)
        f_true_row = f_rec[i, :, h, w].numpy().astype(np.float64)

        for k in SHIFT_RANGE:
            x_shifted = np.roll(x_row, k)
            x_up = np.repeat(x_shifted, haemo_ratio)
            layer = CortexLayer(
                depth=0,
                tau=float(haemo.tau),
                state=HaemodynamicState(x=0.0, s=0.0, f=1.0, v=1.0, q=1.0),
                lambda_d=0.0,
                drain_from=None,
            )
            out = simulate_cortex(
                [layer],
                constants,
                [x_up],
                dt=haemo_dt,
                tau_d=float(haemo.tau_d),
                order=str(full_cfg.model.loss_config.order),
            )
            s_sim = out[0]["s"][::haemo_ratio]
            f_sim = out[0]["f"][::haemo_ratio]
            s_gap = rms(torch.from_numpy(s_sim - s_true_row)) / (
                rms(torch.from_numpy(s_true_row)) + 1e-8
            )
            f_gap = rms(torch.from_numpy(f_sim - f_true_row)) / (
                rms(torch.from_numpy(f_true_row)) + 1e-8
            )
            report_per_shift[k].append((s_gap, f_gap))

    report = {
        f"shift_{k:+d}": {
            "s_mean_rel_rmse": float(np.mean([v[0] for v in vals])),
            "f_mean_rel_rmse": float(np.mean([v[1] for v in vals])),
            "n": len(vals),
        }
        for k, vals in report_per_shift.items()
    }
    write_report(report, diag_out_dir, "LS8_ground_truth_resim_alignment")

    best_shift = min(report, key=lambda k: report[k]["s_mean_rel_rmse"])
    assert best_shift == "shift_+0", (
        f"Re-simulated s (from x_true, no shift) does not best match recorded s_true -- "
        f"best shift was {best_shift}, not 0: {report}"
    )
