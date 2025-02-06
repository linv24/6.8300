import numpy as np
from beartype.typing import List


def create_gaussian_pyramid(video: np.ndarray, num_levels: int = 4) -> List[np.ndarray]:
    """Return a list with Gaussian pyramid of the video. You may find cv2.pyrUp useful."""

    raise NotImplementedError("This is your homework.")


def create_laplacian_pyramid(gaussian_pyramid: List[np.ndarray]) -> List[np.ndarray]:
    """Return a list with Laplacian pyramid of the video. You may find cv2.pyrUp useful."""

    raise NotImplementedError("This is your homework.")


def butter_bandpass_filter(
    laplace_video: np.ndarray,
    low_freq: float = 0.4,
    high_freq: float = 3.0,
    fs: float = 30.0,
    filter_order: int = 5,
) -> np.ndarray:
    """Filter video using a bandpass filter."""
    raise NotImplementedError("This is your homework.")


def filter_laplacian_pyramid(
    laplacian_pyramid: List[np.ndarray],
    fs: float = 30.0,
    low: float = 0.4,
    high: float = 3.0,
    amplification: float = 20.0,
) -> List[np.ndarray]:
    """Filter each level of a Laplacian pyramid using a bandpass filter
    and amplify the result."""

    raise NotImplementedError("This is your homework.")


def create_euler_magnified_video(
    video: np.ndarray, bandpass_filtered: List[np.ndarray]
) -> np.ndarray:
    """Combine all the bandpassed filtered signals to one matrix which is the same
    dimensions as the input video.
    Hint: start from the lowest resolution of the amplified filtered signal,
    upsample that using cv2.pyrUp and add it to the amplified filtered signal
    at the next higher resolution.
    The output video, 'euler_magnified_video', will be the
    input video frames + combined magnified signal."""

    raise NotImplementedError("This is your homework.")
