import numpy as np
from beartype.typing import List

import cv2
from scipy.signal import butter, lfilter

def create_gaussian_pyramid(video: np.ndarray, num_levels: int = 4) -> List[np.ndarray]:
    """Return a list with Gaussian pyramid of the video. You may find cv2.pyrDown useful."""

    G = [video]
    for _ in range(num_levels - 1):
        # downsample each frame individually
        curr_video = G[-1]
        num_frames, height, width, num_channels = curr_video.shape
        downsampled_video = np.empty((num_frames, height // 2, width // 2, num_channels))
        for frame_ix in range(num_frames):
            downsampled_video[frame_ix] = cv2.pyrDown(curr_video[frame_ix])
        G.append(downsampled_video)

    return G


def create_laplacian_pyramid(gaussian_pyramid: List[np.ndarray]) -> List[np.ndarray]:
    """Return a list with Laplacian pyramid of the video. You may find cv2.pyrDown useful."""

    # laplacian computed from the difference of the original image with
    # the upsampled following gaussian pyramid

    L = []
    for g_ix in range(len(gaussian_pyramid) - 1):
        curr_video = gaussian_pyramid[g_ix]
        next_video = gaussian_pyramid[g_ix + 1]

        num_frames, height, width, num_channels = next_video.shape
        upsampled_next_video = np.empty((num_frames, height * 2, width * 2, num_channels))
        for frame_ix in range(num_frames):
            upsampled_next_video[frame_ix] = cv2.pyrUp(next_video[frame_ix])
        L.append(curr_video - upsampled_next_video)

    return L


def butter_bandpass_filter(
    laplace_video: np.ndarray,
    low_freq: float = 0.4,
    high_freq: float = 3.0,
    fs: float = 30.0,
    filter_order: int = 5,
) -> np.ndarray:
    """Filter video using a bandpass filter."""
    b, a = butter(filter_order, (low_freq, high_freq), btype="bandpass", fs=fs)
    filtered_video = lfilter(b, a, laplace_video, axis=0)
    return filtered_video


def filter_laplacian_pyramid(
    laplacian_pyramid: List[np.ndarray],
    fs: float = 30.0,
    low: float = 0.4,
    high: float = 3.0,
    amplification: float = 20.0,
) -> List[np.ndarray]:
    """Filter each level of a Laplacian pyramid using a bandpass filter
    and amplify the result."""

    return [butter_bandpass_filter(L, low_freq=low, high_freq=high, fs=fs) * amplification
            for L in laplacian_pyramid]


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

    reversed_pyramid = bandpass_filtered[::-1]
    for ix, L in enumerate(reversed_pyramid[:-1]):
        upsampled_video = np.array([cv2.pyrUp(frame) for frame in L])
        reversed_pyramid[ix + 1] += upsampled_video

    euler_magnified_video = video + reversed_pyramid[-1]
    return euler_magnified_video