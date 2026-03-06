import torch
import torch.distributed as dist
from torch import nn


class LayerwiseBOLDNormalizer(nn.Module):
    """
    Welford online algorithm for running mean/variance.
    Reduces over (B, L, T) to preserve inter-layer amplitude
    and phase relationships at each (H, W) voxel.
    """

    def __init__(
        self,
        H: int,
        W: int,
        eps: float = 1e-6,
        freeze_after_steps: int = 500,
    ):
        super().__init__()
        self.eps = eps
        self.freeze_after_steps = freeze_after_steps

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

    def _welford_update(self, bold: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Cast to fp32 before accumulation as bold may arrive as fp16 from the
        # dataloader, and computing mean/var in fp16 before writing into fp32
        # buffers loses precision in the variance estimate.
        bold = bold.detach().float()

        B, L, T, H, W = bold.shape
        n_new = B * L * T * H * W

        batch_mean = bold.mean(dim=(0, 1, 2, 3, 4), keepdim=True)
        batch_var = bold.var(dim=(0, 1, 2, 3, 4), keepdim=True, unbiased=False)
        batch_M2 = batch_var * n_new

        # Sync stats across DDP ranks via parallel Welford combination.
        # Requires drop_last=True so all ranks have the same n_new.
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            n_global = n_new * world_size
            # Pack weighted_sum and sum_of_squares into one all-reduce.
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
        new_mean = self.running_mean + delta * n_new / n_combined
        new_M2 = self.running_M2 + batch_M2 + delta**2 * n_old * n_new / n_combined

        self.running_mean.copy_(new_mean)
        self.running_M2.copy_(new_M2)
        self.running_count.add_(n_new)
        self.step.add_(1)

        return batch_mean, batch_var

    def forward(self, bold: torch.Tensor) -> torch.Tensor:
        input_dtype = bold.dtype
        bold_f32 = bold.float()
        if self.training and not self.frozen:
            batch_mean, batch_var = self._welford_update(bold)
            mean = batch_mean
            std = batch_var.sqrt().clamp(min=1e-3)
        else:
            mean = self.running_mean
            std = self.running_var.sqrt().clamp(min=1e-3)

        return ((bold_f32 - mean) / (std + self.eps)).clamp(-10.0, 10.0).to(input_dtype)

    def normalize(self, bold: torch.Tensor) -> torch.Tensor:
        input_dtype = bold.dtype
        std = self.running_var.sqrt().clamp(min=1e-3)
        return (
            ((bold.float() - self.running_mean) / (std + self.eps))
            .clamp(-10.0, 10.0)
            .to(input_dtype)
        )

    def denormalize(self, bold_norm: torch.Tensor) -> torch.Tensor:
        std = self.running_var.sqrt().clamp(min=1e-3)
        return (bold_norm.float() * (std + self.eps) + self.running_mean).to(bold_norm.dtype)
