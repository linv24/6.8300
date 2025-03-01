import torch
import torch.nn.functional as F
from jaxtyping import Float, Complex
from torch import Tensor
import numpy as np

from .gauss import _gaussian_filter_1d


def oriented_filter(theta: float, sigma: float, **kwargs) -> Float[Tensor, "N N"]:
    """
    Return an oriented first-order Gaussian filter
    given an angle (in radians) and standard deviation.

    Hint:
    - Use `.gauss._gaussian_filter_1d`!

    Implementation details:
    - **kwargs are passed to `_gaussian_filter_1d`
    """
    # Get 1D Gaussian filters
    gaussian = _gaussian_filter_1d(sigma, order=0, **kwargs)
    gaussian_prime = _gaussian_filter_1d(sigma, order=1, **kwargs)

    # Compute separable filters for x and y derivatives
    G_x = gaussian[:, None] @ gaussian_prime[None, :]  # First derivative in x, smoothed in y
    G_y = gaussian_prime[:, None] @ gaussian[None, :]  # First derivative in y, smoothed in x

    return G_x * np.cos(theta) + G_y * np.sin(theta)


def conv(
    img: Float[Tensor, "B 1 H W"],  # Input image
    kernel: Float[Tensor, "N N"] | Complex[Tensor, "N N"],  # Convolutional kernel
    mode: str = "reflect",  # Padding mode
) -> Float[Tensor, "B 1 H W"]:
    """
    Convolve an image with an oriented first-order Gaussian filter
    given an angle (in radians) and standard deviation.
    """
    return F.conv2d(img, kernel.unsqueeze(0).unsqueeze(0))


def steer_the_filter(
    img: Float[Tensor, "B 1 H W"], theta: float, sigma: float, **kwargs
) -> Float[Tensor, "B 1 H W"]:
    """
    Return the image convolved with a steered filter.
    """
    kernel = oriented_filter(theta, sigma, **kwargs)
    return conv(img, kernel)


def steer_the_images(
    img: Float[Tensor, "B 1 H W"], theta: float, sigma: float, **kwargs
) -> Float[Tensor, "B 1 H W"]:
    """
    Return the steered image convolved with a filter.
    """
    kernel_x = oriented_filter(0, sigma, **kwargs)
    kernel_y = oriented_filter(np.pi/2, sigma, **kwargs)
    filtered_img_x = conv(img, kernel_x)[None, :]
    filtered_img_y = conv(img, kernel_y)[:, None]

    return filtered_img_x * np.cos(theta) + filtered_img_y * np.sin(theta)


import cv2
def measure_orientation(
    img: Float[Tensor, "B 1 H W"], sigma: float, **kwargs
) -> Float[Tensor, "B 1 H W"]:
    """
    Design a filter to measure the orientation of edges in an image.

    Hint:
    - Consider the complex filter from the README
    - You will need to design a method for noise suppression
    """
    # pi = np.pi
    # thetas = [-pi, -3*pi/4, -pi/2, -pi/4, 0, pi/4, pi/2, 3*pi/4, pi]

    # Get 1D Gaussian filters
    gaussian = _gaussian_filter_1d(sigma, order=0, **kwargs)
    gaussian_prime = _gaussian_filter_1d(sigma, order=1, **kwargs)
    G_x = gaussian[:, None] @ gaussian_prime[None, :]  # First derivative in x, smoothed in y
    G_y = gaussian_prime[:, None] @ gaussian[None, :]  # First derivative in y, smoothed in x

    complex_filter = G_x + 1j * G_y
    denoised_image = cv2.GaussianBlur(img.squeeze().numpy(), (5,5), 0)
    denoised_image = torch.from_numpy(denoised_image).unsqueeze(0).unsqueeze(0)
    complex_img = conv(torch.tensor(denoised_image, dtype=torch.cfloat), complex_filter)

    return np.angle(complex_img)
