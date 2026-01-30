import torch
from torch import nn


class LinearBlock(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        activation: str,
        normalisation: str,
        do_residual: bool,
    ) -> None:
        self.input_dims = input_dims
        self.outut_dims = output_dims
        self.activation = activation
        self.normalisation = normalisation
        self.do_residual = do_residual
        self._init_param()

    def _init_weights(self) -> None:
        pass

    def _init_bias(self) -> None:
        pass

    def _init_param(self) -> None:
        self.linear = nn.Linear(self.input_dims, self.outut_dims)

        self._init_weights()
        self._init_bias()

    def _init_activation(self) -> None:
        pass
