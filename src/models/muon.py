"""Single-GPU Muon optimizer (Newton-Schulz orthogonalization + Nesterov momentum).

Reference: https://kellerjordan.github.io/posts/muon/
"""
from __future__ import annotations
import torch
from torch.optim import Optimizer


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Orthogonalize G via degree-5 Newton-Schulz iteration."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class MuonSingleGPU(Optimizer):
    """Muon for single-GPU / non-distributed training.

    - 2D+ params (weight matrices, conv weights): Nesterov momentum + NS orthogonalization
    - 1D params (biases, norms, embeddings): standard AdamW
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_wd: float = 1e-5,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, ns_steps=ns_steps,
            adamw_lr=adamw_lr, adamw_betas=adamw_betas, adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            adamw_lr = group["adamw_lr"]
            b1, b2 = group["adamw_betas"]
            adamw_wd = group["adamw_wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if p.ndim >= 2:
                    # Muon update
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    # Nesterov: use grad + momentum * buffer
                    g_nes = g.add(buf, alpha=momentum)
                    # Reshape to 2D for orthogonalization
                    shape = g_nes.shape
                    g2d = g_nes.reshape(shape[0], -1)
                    update = _zeropower_via_newtonschulz5(g2d, steps=ns_steps).reshape(shape)
                    # Scale to match RMS of grad
                    scale = max(1, g2d.size(0) / g2d.size(1)) ** 0.5
                    p.add_(update, alpha=-lr * scale)
                else:
                    # AdamW update
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.mul_(b1).add_(g, alpha=1 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1 - b2)
                    bc1 = 1 - b1 ** state["step"]
                    bc2 = 1 - b2 ** state["step"]
                    step = (m / bc1) / ((v / bc2).sqrt() + 1e-8)
                    p.mul_(1 - adamw_lr * adamw_wd)
                    p.add_(step, alpha=-adamw_lr)

        return loss
