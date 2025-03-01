from math import factorial, ceil, sqrt

import torch
from jaxtyping import Float
from torch import Tensor
import numpy as np


def _gaussian_filter_1d(
    sigma: float,  # Standard deviation of the Gaussian
    order: int,  # Order of the derivative
    truncate: float = 4.0,  # Truncate the filter at this many standard deviations
    dtype: torch.dtype = torch.float32,  # Data type to run the computation in
    device: torch.device = torch.device("cpu"),  # Device to run the computation on
) -> Float[Tensor, " filter_size"]:
    """
    Return a 1D Gaussian filter of a specified order.

    Implementation details:
    - filter_size = 2r + 1, where r = ceil(truncate * sigma)
    """
    r = np.ceil(truncate * sigma)
    # ensure kernel is symmetric around 0
    x = torch.arange(-r, r + 1, dtype=dtype, device=device)


    gaussian = torch.exp(-0.5 * (x**2 / sigma**2))
    gaussian /= gaussian.sum() # ensure (weighted) sum of kernel is constant by normalizing

    if order == 0:
        return gaussian
    elif order == 1:
        kernel = -x / sigma**2 * gaussian
        kernel -= kernel.mean() # ensure kernel has zero mean
        return kernel
    elif order == 2:
        kernel = ((x**2 - sigma**2) / sigma**4) * gaussian
        return kernel

