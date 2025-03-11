import os
import sys
import env
import src.utils.utils as utils

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates


def get_3D_object_points(chessboard_size: tuple) -> np.ndarray:
    """
    Get the 3D object points of a chessboard
    Args:
        chessboard_size: Tuple containing the number of columns and rows in the chessboard
    Returns:
        Numpy array containing the 3D object points
    """
    num_cols, num_rows = chessboard_size
    points = np.array([
        [col, row, 0] for row in range(num_rows) for col in range(num_cols)
    ], dtype=np.float32)

    return points


def undistort_image(image: np.ndarray,
                    camera_matrix: np.ndarray,
                    dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Undistort an image
    Args:
        image: Numpy array containing the image
        camera_matrix: Numpy array containing the camera matrix
        dist_coeffs: Numpy array containing the distortion coefficients
    Returns:
        Numpy array containing the undistorted image
    """
    height, width, channels = image.shape
    inv_camera_matrix = np.linalg.inv(camera_matrix)
    k1, k2, p1, p2, k3 = dist_coeffs.flatten()

    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    for v in range(height):
        for u in range(width):
            uv_homogenous = np.array([u, v, 1], dtype=np.float32)
            x, y, _ = inv_camera_matrix @ uv_homogenous # world coordinates

            r = np.sqrt(x**2 + y**2)
            x_distorted = (x * (1 + k1*r**2 + k2*r**4 + k3*r**6)
                           + (2*p1*x*y) + p2 * (r**2 + 2*x**2))
            y_distorted = (y * (1 + k1*r**2 + k2*r**4 + k3*r**6)
                           + (2*p2*x*y) + p1 * (r**2 + 2*y**2))

            u_distorted, v_distorted, _ = camera_matrix @ np.array([x_distorted, y_distorted, 1]) # pixel coordinates
            map_x[v, u] = u_distorted
            map_y[v, u] = v_distorted

    coords = np.stack((map_y.flatten(), map_x.flatten()), axis=0)

    # apply map to each channel individually
    undistorted_image = np.zeros_like(image)
    for c in range(channels):
        mapped_coords = map_coordinates(image[..., c], coords, order=1, mode="nearest").reshape((height, width))
        undistorted_image[..., c] = mapped_coords.reshape((height, width))
    return undistorted_image

def load_grayscale_image(image: np.ndarray) -> np.ndarray:
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    return gray_image


def calibrate_camera(object_points: np.ndarray,
                     corners: np.ndarray,
                     image_size: tuple) -> tuple:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [object_points], [corners], image_size, None, None
    )

    return camera_matrix, dist_coeffs


def find_chessboard_corners(image: np.ndarray, chessboard_size: tuple) -> np.ndarray:
    ret, corners = cv2.findChessboardCorners(image, chessboard_size, None)

    if ret is False:
        raise ValueError("Verify correct dimensions of chessboard")

    return corners


def refine_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)

    return corners


def draw_corners(image: np.ndarray, chessboard_size: tuple, corners: np.ndarray):
    cv2.drawChessboardCorners(image, chessboard_size, corners, True)
    plt.imshow(image)
    plt.title("Chessboard Corners")
    plt.show()


if __name__ == "__main__":
    if not os.path.exists(env.p2.output):
        os.makedirs(env.p2.output)
    expected_camera_matrix = np.load(env.p2.expected_camera_matrix)
    expected_dist_coeffs = np.load(env.p2.expected_dist_coeffs)
    # Part 2.a
    ideal_intrinsic_matrix = np.array([
        [237.9237668337108, 0, 0],
        [0, 237.9237668337108, 0],
        [0, 0, 1]
    ])

    # Part 2.b
    chessboard_size = (14, 9)  # (columns, rows)

    image = utils.load_image(env.p1.chessboard_path)
    grayscale_image = load_grayscale_image(image)
    corners = find_chessboard_corners(grayscale_image, chessboard_size)
    corners = refine_corners(grayscale_image, corners)
    draw_corners(image, chessboard_size, corners)
    Image.fromarray(image).save(env.p2.chessboard_corners)

    # Part 2.c
    object_points = get_3D_object_points(chessboard_size)
    camera_matrix, dist_coeffs = calibrate_camera(object_points, corners, grayscale_image.shape[::-1])
    print("Camera Matrix:")
    print(camera_matrix)
    assert np.allclose(camera_matrix, expected_camera_matrix, atol=1e-2), f"Camera matrix does not match this expected matrix:\n{expected_camera_matrix}"
    np.save(env.p2.camera_matrix, camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs)
    assert np.allclose(dist_coeffs, expected_dist_coeffs, atol=1e-2), f"Distortion coefficients do not match these expected coefficients:\n{expected_dist_coeffs}"
    np.save(env.p2.dist_coeff, dist_coeffs)

    # Part 2.d
    undistorted_image = undistort_image(image, camera_matrix, dist_coeffs)
    plt.imshow(undistorted_image)
    plt.title("Undistorted Image")
    plt.show()
    Image.fromarray(undistorted_image).save(env.p2.undistorted_image)