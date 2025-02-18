import numpy as np
from numpy import pi


def magnify_change(
    im1: np.ndarray, im2: np.ndarray, magnification_factor: int
) -> np.ndarray:
    """Magnify the motion between two images."""
    ft_im1, ft_im2 = np.fft.fft2(im1), np.fft.fft2(im2)

    phase_diff = process_phase_shift(np.angle(ft_im2), np.angle(ft_im1))
    phase_diff *= magnification_factor

    magnified_im = abs(ft_im2) * np.exp((np.angle(ft_im1) + phase_diff) * 1j)
    return np.fft.ifft2(magnified_im).real


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

    expected[0, 4] = 1.0
    expected[4, 8] = 1.0
    return expected


def magnify_motion_local(
    im1: np.ndarray, im2: np.ndarray, magnification_factor: int, sigma: int
) -> np.ndarray:
    """Magnify motion using localized processing with Gaussian windows."""
    im_size = im1.shape[0]
    magnified = np.zeros([im_size, im_size])

    ''' provided pseudocode:
    X, Y = np.meshgrid(np.arange(im_size), np.arange(im_size))
    for y in range(0, im_size, 2 * sigma):
        for x in range(0, im_size, 2 * sigma):
            # TODO: Create a Gaussian mask that covers the whole image and apply
            # it to the images
            # TODO: Magnify the phase changes

    return magnified_image
    '''

    X, Y = np.meshgrid(np.arange(im_size), np.arange(im_size))
    for y in range(0, im_size, 2 * sigma):
        for x in range(0, im_size, 2 * sigma):
            gaussian_mask = np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))
            masked_im1 = im1 * gaussian_mask
            masked_im2 = im2 * gaussian_mask

            magnified_im = magnify_change(masked_im1, masked_im2, magnification_factor)
            magnified += magnified_im

    return magnified


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

    ''' provided pseudocode:
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    for y in range(0, height, 2 * sigma):
        for x in range(0, width, 2 * sigma):
            # TODO: Create a Gaussian mask that covers the whole image

            for channel_index in range(num_channels):
                # initialize a moving average
                window_avg_phase = None

                for frame_index in range(num_frames):
                    #TODO: Apply gaussian mask to frame

                    #TODO: Perform magnification

                    #TODO: # Aggregate this window's contribution.

    return magnified
    '''

    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    for y in range(0, height, 2 * sigma):
        for x in range(0, width, 2 * sigma):
            gaussian_mask = np.exp(-((X - x)**2 + (Y - y)**2) / (2 * sigma**2))

            for channel_ix in range(num_channels):
                # initialize a moving average
                window_avg_phase = None

                for frame_ix in range(num_frames):
                    if frame_ix == 0:
                        magnified[frame_ix, ..., channel_ix] = frames[frame_ix, ..., channel_ix]
                        continue
                    masked_curr_frame = frames[frame_ix, ..., channel_ix] * gaussian_mask
                    masked_prev_frame = frames[frame_ix - 1, ..., channel_ix] * gaussian_mask

                    fft_curr_frame = np.fft.fft2(masked_curr_frame)
                    fft_prev_frame = np.fft.fft2(masked_prev_frame)

                    phase_shift = process_phase_shift(np.angle(fft_curr_frame), np.angle(fft_prev_frame))
                    phase_shift *= magnification_factor

                    if window_avg_phase is None:
                        window_avg_phase = phase_shift
                    else:
                        window_avg_phase = update_moving_average(window_avg_phase, phase_shift, alpha)

                    magnified_frame = abs(fft_curr_frame) * np.exp((np.angle(fft_prev_frame) + window_avg_phase) * 1j)
                    magnified[frame_ix, ..., channel_ix] += np.fft.ifft2(magnified_frame).real

    return magnified