import numpy as np
from numpy import pi


def magnify_change(
    im1: np.ndarray, im2: np.ndarray, magnification_factor: int
) -> np.ndarray:
    """Magnify the motion between two images."""
    raise NotImplementedError("This is your homework.")


def magnify_motion_global_question() -> np.ndarray:
    """
    Given two 9x9 images with the following point movements:
    1. A point moves from position (0,0) to (0,1)
    2. A point moves from position (8,8) to (7,8)

    If we magnify this motion by 4x using phase-based motion magnification,
    what positions would you expect the points to move to?

    Fill in the expected matrix to show where the points should appear
    after 4x magnification.
    """
    im_size = 9
    im1 = np.zeros([im_size, im_size], dtype=np.float32)
    im2 = np.zeros([im_size, im_size], dtype=np.float32)

    # Initial positions
    im1[0, 0] = 1.0
    im2[0, 1] = 1.0
    im1[8, 8] = 1.0
    im2[7, 8] = 1.0

    # Fill in your expected output matrix
    expected = np.zeros([im_size, im_size], dtype=np.float32)

    raise NotImplementedError("This is your homework.")


def magnify_motion_local(
    im1: np.ndarray, im2: np.ndarray, magnification_factor: int, sigma: int
) -> np.ndarray:
    """Magnify motion using localized processing with Gaussian windows."""
    im_size = im1.shape[0]
    magnified = np.zeros([im_size, im_size])

    raise NotImplementedError("This is your homework.")


def process_phase_shift(
    current_phase: np.ndarray, reference_phase: np.ndarray
) -> np.ndarray:
    """Computes phase shift and constrains it to [-π, π]"""
    shift = current_phase - reference_phase
    shift[shift > pi] -= 2 * pi
    shift[shift < -pi] += 2 * pi
    return shift


def update_moving_average(
    prev_average: np.ndarray, new_value: np.ndarray, alpha: float
) -> np.ndarray:
    """Updates the moving average of phase with temporal smoothing"""
    return alpha * prev_average + (1 - alpha) * new_value


def magnify_motion_video(
    frames: np.ndarray, magnification_factor: int, sigma: int, alpha: float
) -> np.ndarray:
    """Magnifies subtle motions in the video frames using phase-based magnification."""
    num_frames, height, width, num_channels = frames.shape
    magnified = np.zeros_like(frames)

    raise NotImplementedError("This is your homework.")
