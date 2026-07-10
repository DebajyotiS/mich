import torch
import torch.distributed as dist
from torch import nn


class LayerwiseBOLDNormalizer(nn.Module):
    """
    Welford online algorithm for running mean/variance.

    Statistics are computed from a spatial neighbourhood of radius
    `neighbourhood_radius` around each sample's active source voxels,
    reducing over (valid sources, L, T, N) to a single shared scalar. This
    excludes the mostly-silent background which would otherwise dominate
    the variance estimate, while preserving inter-layer amplitude ratios
    (all layers divided by the same scale factor).
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

    @staticmethod
    def _source_mask(num_sources: torch.Tensor, S: int) -> torch.Tensor:
        """[B] valid-source counts -> [B, S] boolean mask (arange trick, not a padding-value check)."""
        arange_s = torch.arange(S, device=num_sources.device)
        return arange_s[None, :] < num_sources[:, None]

    def _gather_neighbourhood(
        self, bold: torch.Tensor, source_position: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Gather voxels within a square neighbourhood around each valid source.

        bold:            [B, L, T, H, W]
        source_position: [B, S, 2]  (h, w) -- padded rows may hold garbage, excluded via `mask`
        mask:            [B, S]     bool, True where that source slot is valid
        returns:         [M, L, T, N]  where N = (2r+1)^2, M = mask.sum()
        """
        B, L, T, H, W = bold.shape
        S = source_position.shape[1]
        r = self.neighbourhood_radius
        device = bold.device

        offsets = torch.arange(-r, r + 1, device=device)
        oh, ow = torch.meshgrid(offsets, offsets, indexing="ij")
        oh = oh.reshape(-1)  # [N]
        ow = ow.reshape(-1)  # [N]

        src_h = source_position[..., 0].long()  # [B, S]
        src_w = source_position[..., 1].long()  # [B, S]

        nh = (src_h[..., None] + oh[None, None]).clamp(0, H - 1)  # [B, S, N]
        nw = (src_w[..., None] + ow[None, None]).clamp(0, W - 1)  # [B, S, N]

        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1, 1)
        l_idx = torch.arange(L, device=device).view(1, 1, L, 1, 1)
        t_idx = torch.arange(T, device=device).view(1, 1, 1, T, 1)
        nh_ = nh.view(B, S, 1, 1, -1)
        nw_ = nw.view(B, S, 1, 1, -1)

        neighbourhood = bold[b_idx, l_idx, t_idx, nh_, nw_]  # [B, S, L, T, N]
        neighbourhood = neighbourhood.reshape(B * S, L, T, -1)
        return neighbourhood[mask.reshape(-1)]  # [M, L, T, N]

    def _welford_update(
        self, bold: torch.Tensor, source_position: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bold = bold.detach().float()

        neighbourhood = self._gather_neighbourhood(bold, source_position, mask)  # [M, L, T, N]

        # Reduce over all of (M, L, T, N) -- shared scalar preserves inter-layer ratios
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
        num_sources: torch.Tensor | None = None,
        pause_update: bool = False,
    ) -> torch.Tensor:
        input_dtype = bold.dtype
        bold_f32 = bold.float()

        if self.training and (not self.frozen) and (not pause_update):
            if source_position is None or num_sources is None:
                raise ValueError(
                    "source_position and num_sources are required during training "
                    "for neighbourhood normalisation"
                )
            mask = self._source_mask(num_sources, source_position.shape[1])
            batch_mean, batch_var = self._welford_update(bold, source_position, mask)
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
