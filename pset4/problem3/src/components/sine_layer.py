import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.d_in = d_in
        raise NotImplementedError("Use your problem1 implementation")

    def init_weights(self):
        # first layer: sin(w_0 * Wx + b)
        # general: sin(w^Tx + b)
        a, b =  -np.sqrt(6 / self.d_in), np.sqrt(6 / self.d_in)
        self.linear.weight.uniform_(a, b)
        if self.is_first:
            # scale by factor of omega_0
            self.linear.weight *= self.omega_0

    def forward(self, input):
        return torch.sin(self.linear(input))
