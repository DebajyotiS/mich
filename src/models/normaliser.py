import torch
import torch.distributed as dist
from torch import nn


class LayerwiseBOLDNormalizer(nn.Module):
    """
    Welford online algorithm for running mean/variance.

    Statistics are computed from a spatial neighbourhood of radius
    `neighbourhood_radius` around each sample's source voxel, reducing over
    (B, L, T, N) to a single shared scalar. This excludes the mostly-silent
    background which would otherwise dominate the variance estimate, while
    preserving inter-layer amplitude ratios (all layers divided by the same
    scale factor).
    """

    def __init__(
        self,
        H: int,
        W: int,
        eps: float = 1e-6,
        freeze_after_steps: int = 5000,
        neighbourhood_radius: int = 5,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.eps = eps
        self.freeze_after_steps = freeze_after_steps
        self.neighbourhood_radius = neighbourhood_radius

        self.register_buffer("running_mean", torch.zeros(1, 1, 1, 1, 1))
        self.register_buffer("running_M2", torch.zeros(1, 1, 1, 1, 1))
        self.register_buffer("running_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    @property
    def frozen(self) -> bool:
        return self.step.item() >= self.freeze_after_steps

    @property
    def running_var(self) -> torch.Tensor:
        if self.running_count < 2:
            return torch.ones_like(self.running_M2)
        return self.running_M2 / (self.running_count - 1)

    def _gather_neighbourhood(
        self, bold: torch.Tensor, source_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Gather voxels within a square neighbourhood around each sample's source.

        bold:            [B, L, T, H, W]
        source_position: [B, 2]  (h, w)
        returns:         [B, L, T, N]  where N = (2r+1)^2
        """
        B, L, T, H, W = bold.shape
        r = self.neighbourhood_radius
        device = bold.device

        offsets = torch.arange(-r, r + 1, device=device)
        oh, ow = torch.meshgrid(offsets, offsets, indexing="ij")
        oh = oh.reshape(-1)  # [N]
        ow = ow.reshape(-1)  # [N]

        src_h = source_position[:, 0].long()  # [B]
        src_w = source_position[:, 1].long()  # [B]

        nh = (src_h[:, None] + oh[None]).clamp(0, H - 1)  # [B, N]
        nw = (src_w[:, None] + ow[None]).clamp(0, W - 1)  # [B, N]

        return bold[
            torch.arange(B, device=device)[:, None, None, None],
            torch.arange(L, device=device)[None, :, None, None],
            torch.arange(T, device=device)[None, None, :, None],
            nh[:, None, None, :],
            nw[:, None, None, :],
        ]  # [B, L, T, N]

    def _welford_update(
        self, bold: torch.Tensor, source_position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bold = bold.detach().float()
        B, L, T, H, W = bold.shape

        neighbourhood = self._gather_neighbourhood(bold, source_position)  # [B, L, T, N]

        # Reduce over all of (B, L, T, N) — shared scalar preserves inter-layer ratios
        n_new = neighbourhood.numel()
        batch_mean = neighbourhood.mean().reshape(1, 1, 1, 1, 1)
        batch_var = neighbourhood.var(unbiased=False).reshape(1, 1, 1, 1, 1)
        batch_M2 = batch_var * n_new

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            n_global = n_new * world_size
            packed = torch.cat([batch_mean * n_new, batch_M2 + n_new * batch_mean**2], dim=0)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            mean_global = packed[:1] / n_global
            M2_global = (packed[1:] - n_global * mean_global**2).clamp(min=0)
            n_new = n_global
            batch_mean = mean_global
            batch_M2 = M2_global
            batch_var = M2_global / n_new

        n_old = self.running_count.item()
        n_combined = n_old + n_new

        delta = batch_mean - self.running_mean
        self.running_mean.copy_(self.running_mean + delta * n_new / n_combined)
        self.running_M2.copy_(self.running_M2 + batch_M2 + delta**2 * n_old * n_new / n_combined)
        self.running_count.add_(n_new)
        self.step.add_(1)

        return batch_mean, batch_var

    def forward(
        self,
        bold: torch.Tensor,
        source_position: torch.Tensor | None = None,
        pause_update: bool = False,
    ) -> torch.Tensor:
        input_dtype = bold.dtype
        bold_f32 = bold.float()

        if self.training and (not self.frozen) and (not pause_update):
            if source_position is None:
                raise ValueError(
                    "source_position required during training for neighbourhood normalisation"
                )
            batch_mean, batch_var = self._welford_update(bold, source_position)
            mean = batch_mean
            std = batch_var.sqrt().clamp(min=1e-3)
        else:
            mean = self.running_mean
            std = self.running_var.sqrt().clamp(min=1e-3)

        return ((bold_f32 - mean) / std).clamp(-10.0, 10.0).to(input_dtype)

    def normalize(self, bold: torch.Tensor) -> torch.Tensor:
        input_dtype = bold.dtype
        std = self.running_var.sqrt().clamp(min=1e-3)
        return ((bold.float() - self.running_mean) / std).clamp(-10.0, 10.0).to(input_dtype)

    def denormalize(self, bold_norm: torch.Tensor) -> torch.Tensor:
        std = self.running_var.sqrt().clamp(min=1e-3)
        return (bold_norm.float() * std + self.running_mean).to(bold_norm.dtype)
