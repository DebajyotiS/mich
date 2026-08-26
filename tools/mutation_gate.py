#!/usr/bin/env python
"""Mutation gate: check that the test suite still *detects* a curated catalogue of
defects in the science-critical physics code.

Coverage measures which lines run. It says nothing about whether a test would
notice if those lines were wrong. This script closes that gap for a fixed set of
defects: it introduces each one into the source, runs the tests that should catch
it, and fails if the suite stays green.

Every mutation here was verified to survive the suite as it stood before the
oracles in `tests/` were added, and to be caught after. The gate therefore
protects those oracles: if someone later weakens or deletes one, the
corresponding mutation starts surviving and this fails.

Usage:
    python tools/mutation_gate.py                # run the whole catalogue
    python tools/mutation_gate.py --list         # show it without running
    python tools/mutation_gate.py -k balloon     # run a subset by id substring
    python tools/mutation_gate.py -q             # only report failures

Exit status is 0 only when every mutation was caught.

Safety
------
Mutations are applied to the real files in `src/` and reverted with
`git checkout --`, which is authoritative and also refreshes the file mtime (see
*Bytecode* below). To make that revert safe the script refuses to start unless
`src/` is free of uncommitted changes, so it can never destroy local work, and it
restores everything from an `atexit` hook plus SIGINT/SIGTERM handlers so an
interrupted run does not leave mutated source behind.

Bytecode
--------
`__pycache__` is cleared before and after every run. A `.pyc` records its
source's mtime at one-second resolution, so a mutation that preserves file size
(`/ 2` -> `/ 4`, swapping two same-length operands) applied and reverted inside
one second can leave Python executing the *mutated* bytecode afterwards. That
failure mode silently turns a real gap into an apparent pass, which is precisely
the direction a gate must not get wrong.

Serial by design
----------------
Mutations share one working tree, so they cannot run concurrently. The whole
catalogue takes roughly two minutes. Each pytest subprocess is pinned to two
threads so a run does not saturate a shared machine.
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
TESTS = REPO / "tests"


@dataclass(frozen=True)
class Mutation:
    """One defect to inject.

    Attributes:
        id: Stable identifier, `module:short-slug`, used by `-k` and in output.
        path: Source file to mutate, relative to the repo root.
        old: Exact text to replace. Must occur EXACTLY ONCE in the file; any
            other count is a hard error rather than a skip, so that a catalogue
            entry which has drifted out of date is loud instead of silently
            vacuous.
        new: Replacement text.
        tests: Test paths to run, relative to the repo root.
        why: What the mutation breaks, and which oracle is expected to catch it.
    """

    id: str
    path: str
    old: str
    new: str
    tests: tuple[str, ...]
    why: str


# ---------------------------------------------------------------------------
# The catalogue.
#
# Grouped by module. Every entry's `why` names the oracle that catches it, so a
# failure here points straight at the test that regressed rather than just
# saying "a mutant survived".
# ---------------------------------------------------------------------------

L = "src/mich/models/mich_losses.py"
BA = "src/mich/data/balloon.py"
NE = "src/mich/data/neuronal.py"
BL = "src/mich/models/blocks.py"
CO = "src/mich/models/collocation.py"
NR = "src/mich/models/normaliser.py"

TL = ("tests/test_mich_losses.py",)
TBA = ("tests/test_balloon.py",)
TNE = ("tests/test_neuronal.py",)
TBL = ("tests/test_blocks.py",)
TCO = ("tests/test_collocation.py",)
TNR = ("tests/test_normaliser.py",)

CATALOGUE: list[Mutation] = [
    # -- mich_losses: the torch-side physics loss -----------------------------
    Mutation(
        "losses:bold-k2-k3",
        L,
        "return V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))",
        "return V0 * (k1 * (1 - q) + k3 * (1 - q / v) + k2 * (1 - v))",
        TL,
        "Permutes the BOLD readout terms. Caught by the hand-computed "
        "_compute_bold value oracle, which needs k1/k2/k3 all distinct.",
    ),
    Mutation(
        "losses:src-term-sign",
        L,
        "total = colloc_loss + self.hparams.loss_config.lambda_src * src_loss",
        "total = colloc_loss - self.hparams.loss_config.lambda_src * src_loss",
        TL,
        "Flips the source loss into a reward. Caught by the documented "
        "total == colloc + lambda_src * src contract, which needs lambda_src != 1.",
    ),
    Mutation(
        "losses:s-ode-kappa-gamma-swap",
        L,
        "s_target = x - kappa * s - gamma * (f - 1)",
        "s_target = x - gamma * s - kappa * (f - 1)",
        TL,
        "Swaps two physiological constants in the flow-inducing ODE. Caught by "
        "the ODE-residual-is-zero oracle.",
    ),
    Mutation(
        "losses:s-ode-kappa-sign",
        L,
        "s_target = x - kappa * s - gamma * (f - 1)",
        "s_target = x + kappa * s - gamma * (f - 1)",
        TL,
        "Turns vasodilatory decay into growth. Caught by the ODE-residual oracle.",
    ),
    Mutation(
        "losses:v-target-tau",
        L,
        "target_vdot[:, burn_in:] / tau,",
        "target_vdot[:, burn_in:] * tau,",
        TL,
        "Multiplies instead of divides by tau. Invisible while tau == 1; caught "
        "by the ODE-residual oracle plus the explicit tau-scaling test.",
    ),
    Mutation(
        "losses:vstar-uses-tau",
        L,
        "v_star_target = (-v_star + v - 1) / tau_d",
        "v_star_target = (-v_star + v - 1) / tau",
        TL,
        "Uses the wrong time constant for the delay filter. Needs tau != tau_d "
        "in the fixture. Caught by the drain-mode ODE-residual oracle.",
    ),
    Mutation(
        "losses:f-ode-target",
        L,
        "f_loss = self._ode_loss_fn(df_dt[:, burn_in:], s[:, burn_in:])",
        "f_loss = self._ode_loss_fn(df_dt[:, burn_in:], x[:, burn_in:])",
        TL,
        "df/dt = s becomes df/dt = x. Caught by the ODE-residual oracle.",
    ),
    Mutation(
        "losses:drain-coupling-sign",
        L,
        "target_vdot = target_vdot + lambda_d * vstar_deeper",
        "target_vdot = target_vdot - lambda_d * vstar_deeper",
        TL,
        "Reverses inter-layer drainage. Caught by the drain-mode ODE-residual oracle at layer 1.",
    ),
    Mutation(
        "losses:qstar-offset",
        L,
        "q_star_target = (-q_star + q - 1) / tau_d",
        "q_star_target = (-q_star + q + 1) / tau_d",
        TL,
        "Shifts the delay filter's resting point. Caught by the drain-mode ODE-residual oracle.",
    ),
    # -- balloon: the numpy-side forward simulator ---------------------------
    Mutation(
        "balloon:grubb-exponent",
        BA,
        "outflow = v ** (1.0 / c.alpha)",
        "outflow = v ** c.alpha",
        TBA,
        "Inverts Grubb's exponent. Vanishes at the resting state; caught by the "
        "off-rest outflow test and the linear-vs-exact convergence rate.",
    ),
    Mutation(
        "balloon:kappa-gamma-swap",
        BA,
        "ds = x - c.kappa * s - c.gamma * (f - 1.0)",
        "ds = x - c.gamma * s - c.kappa * (f - 1.0)",
        TBA,
        "Swaps kappa and gamma. Caught by the hand-computed off-rest ds/dt value.",
    ),
    Mutation(
        "balloon:drop-tau-exact",
        BA,
        "dv = (f - outflow) / layer.tau",
        "dv = (f - outflow)",
        TBA,
        "Drops the 1/tau scaling (exact order). Caught by the tau-scaling test.",
    ),
    Mutation(
        "balloon:drop-tau-linear",
        BA,
        "dv = (f - v / c.alpha) / layer.tau",
        "dv = (f - v / c.alpha)",
        TBA,
        "Drops the 1/tau scaling (linear order). Caught by the tau-scaling test.",
    ),
    Mutation(
        "balloon:f-ode-target",
        BA,
        "df = s  # type: ignore[assignment]",
        "df = x  # type: ignore[assignment]",
        TBA,
        "df/dt = s becomes df/dt = x. Caught by the exact df/dt value test.",
    ),
    Mutation(
        "balloon:quad-gamma-coeff",
        BA,
        "gamma = beta * np.log(1 - c.E0) / 2",
        "gamma = beta * np.log(1 - c.E0) / 4",
        TBA,
        "Halves the quadratic f^2 coefficient. Survives a 'quadratic beats "
        "linear' check; caught only by the O(eps^3) convergence rate.",
    ),
    Mutation(
        "balloon:quad-vq-cross",
        BA,
        "- (1 / c.alpha - 1) * v * q",
        "- (1 / c.alpha - 1) * v * f",
        TBA,
        "Wrong pair in the quadratic cross term. Caught by the O(eps^3) rate.",
    ),
    Mutation(
        "balloon:quad-v2-coeff",
        BA,
        "- (1 / 2) * (1 / c.alpha - 1) * (1 / c.alpha - 2) * v**2",
        "- (1 / 2) * (1 / c.alpha - 1) * (1 / c.alpha - 3) * v**2",
        TBA,
        "Perturbs the quadratic v^2 coefficient. Caught by the O(eps^3) rate.",
    ),
    Mutation(
        "balloon:quad-dv-v2-coeff",
        BA,
        "dv = (f - v / c.alpha - (1 - c.alpha) / (2 * c.alpha**2) * v**2) / layer.tau",
        "dv = (f - v / c.alpha - (1 - c.alpha) / (3 * c.alpha**2) * v**2) / layer.tau",
        TBA,
        "Perturbs the quadratic dv/dt v^2 coefficient. Caught by the O(eps^3) rate.",
    ),
    # -- neuronal: the reaction-diffusion PDE --------------------------------
    Mutation(
        "neuronal:decay-sign",
        NE,
        "new_grid[layer_index] -= self.decay_rate * self.grid[layer_index] * dt_sub",
        "new_grid[layer_index] += self.decay_rate * self.grid[layer_index] * dt_sub",
        TNE,
        "Decay becomes growth. Caught by the discrete exponential decay oracle.",
    ),
    Mutation(
        "neuronal:decay-drop-dt",
        NE,
        "new_grid[layer_index] -= self.decay_rate * self.grid[layer_index] * dt_sub",
        "new_grid[layer_index] -= self.decay_rate * self.grid[layer_index]",
        TNE,
        "Drops dt from the decay step. Caught by the exact geometric-ratio oracle.",
    ),
    Mutation(
        "neuronal:laplacian-centre",
        NE,
        "[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]",
        "[[0.0, 1.0, 0.0], [1.0, -3.0, 1.0], [0.0, 1.0, 0.0]]",
        TNE,
        "Breaks the zero-sum stencil, so diffusion creates mass. Caught by the "
        "mass-conservation oracle.",
    ),
    Mutation(
        "neuronal:laplacian-anisotropic",
        NE,
        "[[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]",
        "[[0.0, 1.0, 0.0], [2.0, -4.0, 1.0], [0.0, 1.0, 0.0]]",
        TNE,
        "Makes diffusion direction-dependent. Caught by the isotropy oracle.",
    ),
    Mutation(
        "neuronal:snr-db-convention",
        NE,
        "snr_linear = 10.0 ** (float(snr_db) / 10.0)",
        "snr_linear = 10.0 ** (float(snr_db) / 20.0)",
        TNE,
        "Swaps the power dB convention for the amplitude one. Indistinguishable "
        "at snr_db in {0, inf}; caught by the snr_db=20 case.",
    ),
    Mutation(
        "neuronal:noise-amp-sqrt",
        NE,
        "noise_amp_temporal = np.sqrt(Pn_total * tf)",
        "noise_amp_temporal = Pn_total * tf",
        TNE,
        "Confuses noise power with amplitude. Caught by the SNR conversion "
        "oracle, which needs a noise double that scales by the amplitude it is "
        "handed rather than returning zeros.",
    ),
    Mutation(
        "neuronal:inter-layer-flux-sign",
        NE,
        "* (self.grid[layer_index - 1] - self.grid[layer_index])",
        "* (self.grid[layer_index] - self.grid[layer_index - 1])",
        TNE,
        "Reverses inter-layer flux, so activity flows uphill. Caught by the "
        "inter-layer coupling oracle.",
    ),
    Mutation(
        "neuronal:diffusion-sign",
        NE,
        "new_grid[layer_index] += (self.diff_intra * lap) * dt_sub",
        "new_grid[layer_index] -= (self.diff_intra * lap) * dt_sub",
        TNE,
        "Anti-diffusion. Caught by the mass-conservation and isotropy oracles.",
    ),
    # -- blocks: network building blocks ------------------------------------
    Mutation(
        "blocks:drop-layer-mask",
        BL,
        "W_eff = self.W * self.mask",
        "W_eff = self.W",
        TBL,
        "Removes the causal layer-mixing mask entirely. Invisible while W is "
        "identity; caught by the non-identity-W coupling tests.",
    ),
    Mutation(
        "blocks:fourier-two-pi",
        BL,
        "ang = (2.0 * torch.pi) * t[..., None] * freqs",
        "ang = t[..., None] * freqs",
        TBL,
        "Drops the 2*pi so integer frequencies are no longer periodic in t. "
        "Caught by the periodicity oracle.",
    ),
    Mutation(
        "blocks:fourier-sin-cos-order",
        BL,
        "emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)",
        "emb = torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1)",
        TBL,
        "Swaps the embedding halves. Caught by the emb(0) == [0..., 1...] oracle.",
    ),
    Mutation(
        "blocks:mask-below-to-above",
        BL,
        "mask[idx[1:], idx[:-1]] = 1.0",
        "mask[idx[:-1], idx[1:]] = 1.0",
        TBL,
        "Opens coupling from the layer above instead of below, reversing the "
        "drainage direction. Caught by the two directional mask tests.",
    ),
    # -- collocation: index sampling ----------------------------------------
    Mutation(
        "colloc:full-grid-time-order",
        CO,
        "t = torch.arange(T, device=device).view(1, 1, T, 1).expand(1, 1, T, H * W)",
        "t = (T - 1 - torch.arange(T, device=device)).view(1, 1, T, 1).expand(1, 1, T, H * W)",
        TCO,
        "Reverses chronological order, so downstream burn_in drops the wrong "
        "end. Survives any set-based coverage check; caught by the time-axis "
        "ordering oracle.",
    ),
    Mutation(
        "colloc:src-h-reads-w",
        CO,
        "            src_h = source_position[b_idx, src_choice, 0].long()  "
        "# [B, n_times, n_dense_s]",
        "            src_h = source_position[b_idx, src_choice, 1].long()  "
        "# [B, n_times, n_dense_s]",
        TCO,
        "Reads the w channel as h. Invisible with symmetric source positions "
        "like (10, 10); caught now that positions are asymmetric.",
    ),
    # -- normaliser ---------------------------------------------------------
    Mutation(
        "norm:training-uses-running-stats",
        NR,
        "            mean = batch_mean\n            std = batch_var.sqrt().clamp(min=1e-3)",
        "            mean = self.running_mean\n"
        "            std = self.running_var.sqrt().clamp(min=1e-3)",
        TNR,
        "Breaks the module's central documented semantic (train normalises with "
        "this batch's own statistics). Caught by the differential oracle, which "
        "needs a large running_count so blending cannot mask the difference.",
    ),
    Mutation(
        "norm:batch-std-sqrt",
        NR,
        "std = batch_var.sqrt().clamp(min=1e-3)",
        "std = batch_var.clamp(min=1e-3)",
        TNR,
        "Divides by variance instead of standard deviation. Caught by the "
        "rescaling-invariance oracle.",
    ),
    Mutation(
        "norm:swap-src-h-w",
        NR,
        "        src_h = source_position[..., 0].long()  # [B, S]\n"
        "        src_w = source_position[..., 1].long()  # [B, S]",
        "        src_h = source_position[..., 1].long()  # [B, S]\n"
        "        src_w = source_position[..., 0].long()  # [B, S]",
        TNR,
        "Transposes the neighbourhood gather. Invisible on a square grid with a "
        "symmetric position; caught by the asymmetric-position oracle.",
    ),
    Mutation(
        "norm:denorm-mean-std-swap",
        NR,
        "return (bold_norm.float() * std + self.running_mean).to(bold_norm.dtype)",
        "return (bold_norm.float() * self.running_mean + std).to(bold_norm.dtype)",
        TNR,
        "Swaps the roles of mean and std when inverting. Caught by the exact "
        "affine denormalise oracle, which needs mean != std.",
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class StaleCatalogueError(RuntimeError):
    """A catalogue entry's `old` text no longer matches the source exactly once."""


_dirty: set[str] = set()


def _clear_pycache() -> None:
    for root in (SRC, TESTS):
        for cache in root.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)


def _restore(rel_path: str) -> None:
    subprocess.run(["git", "checkout", "--", rel_path], cwd=REPO, check=True, capture_output=True)
    _dirty.discard(rel_path)


def _restore_all() -> None:
    for rel_path in list(_dirty):
        try:
            _restore(rel_path)
        except subprocess.CalledProcessError:  # pragma: no cover - best effort
            print(f"  !! could not restore {rel_path}; run: git checkout -- {rel_path}")
    if _dirty:
        _clear_pycache()


def _apply(mut: Mutation) -> None:
    target = REPO / mut.path
    text = target.read_text()
    found = text.count(mut.old)
    if found != 1:
        raise StaleCatalogueError(
            f"{mut.id}: pattern occurs {found} times in {mut.path}, expected exactly 1.\n"
            f"  The source has moved on. Update the catalogue entry (or delete it if the "
            f"code it targeted is gone) rather than leaving it silently inert."
        )
    _dirty.add(mut.path)
    target.write_text(text.replace(mut.old, mut.new, 1))


def _run_tests(mut: Mutation) -> bool:
    """Run the guarding tests against mutated source. True if they caught it."""
    env = dict(os.environ)
    # Pin threads so a run stays polite on a shared machine, and keep the
    # interpreter from writing bytecode we would then have to invalidate.
    env["OMP_NUM_THREADS"] = env.get("MUTATION_GATE_THREADS", "2")
    env["MKL_NUM_THREADS"] = env["OMP_NUM_THREADS"]
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *mut.tests],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    # Non-zero exit means at least one test failed, i.e. the defect was detected.
    return proc.returncode != 0


def _preflight() -> None:
    if not (REPO / ".git").exists():
        sys.exit("error: not a git repository; this script restores via `git checkout --`.")
    dirty = subprocess.run(
        ["git", "diff", "--name-only", "--", "src/"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    if dirty:
        sys.exit(
            "error: uncommitted changes under src/:\n"
            + "".join(f"    {p}\n" for p in dirty)
            + "This script reverts mutations with `git checkout --`, which would discard\n"
            "them. Commit or stash first."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-k", dest="filter", help="only run entries whose id contains this")
    parser.add_argument("--list", action="store_true", help="print the catalogue and exit")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print failures")
    args = parser.parse_args()

    selected = [m for m in CATALOGUE if not args.filter or args.filter in m.id]
    if not selected:
        sys.exit(f"error: no catalogue entry matches {args.filter!r}")

    if args.list:
        for mut in selected:
            print(f"{mut.id}\n    {mut.path}\n    {mut.why}\n")
        print(f"{len(selected)} mutation(s)")
        return 0

    _preflight()
    atexit.register(_restore_all)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: sys.exit(130))

    survived: list[Mutation] = []
    stale: list[str] = []
    width = max(len(m.id) for m in selected)

    print(f"Mutation gate: {len(selected)} mutation(s), serial\n")
    for i, mut in enumerate(selected, 1):
        try:
            _apply(mut)
        except StaleCatalogueError as exc:
            stale.append(str(exc))
            print(f"[{i:>2}/{len(selected)}] {mut.id:<{width}}  STALE")
            continue
        try:
            _clear_pycache()
            caught = _run_tests(mut)
        finally:
            _restore(mut.path)
            _clear_pycache()

        if caught:
            if not args.quiet:
                print(f"[{i:>2}/{len(selected)}] {mut.id:<{width}}  caught")
        else:
            survived.append(mut)
            print(f"[{i:>2}/{len(selected)}] {mut.id:<{width}}  SURVIVED")

    print()
    if stale:
        print(f"{len(stale)} stale catalogue entr{'y' if len(stale) == 1 else 'ies'}:\n")
        for msg in stale:
            print(f"  {msg}\n")
    if survived:
        print(f"{len(survived)} mutation(s) went undetected:\n")
        for mut in survived:
            print(f"  {mut.id}  ({mut.path})")
            print(f"      {mut.why}\n")
        print(
            "A surviving mutation means the oracle that used to catch it no longer\n"
            "does. Either restore the assertion it relied on, or, if the behaviour\n"
            "was changed deliberately, update the catalogue entry to match."
        )
        return 1
    if stale:
        return 1

    print(f"All {len(selected)} mutation(s) detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
