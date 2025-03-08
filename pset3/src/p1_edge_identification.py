import os
import sys
import env
import src.utils.engine as engine

import numpy as np
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt


BINARIZE_THRESHOLD = 128

def find_contours(binary_image: np.ndarray, foreground: int=0) -> np.ndarray:
    """
    Find the boundaries of objects in a binary image.
    Args:
        binary_image: A binary image with objects as foreground.
        foreground: The value of the foreground pixels.
    Returns:
        A list of pixel coordinates that form the boundaries of the objects.
    """
    # use Sobel operator for edge detection
    def convolve(image, kernel):
        rows, cols = image.shape
        output = np.zeros_like(image)

        k_size = kernel.shape[0] // 2
        for i in range(k_size, rows - k_size):
            for j in range(k_size, cols - k_size):
                window = image[i-k_size:i+k_size+1, j-k_size:j+k_size+1]
                output[i, j] = abs(np.sum(window * kernel))  # Sum weighted region

        return output

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])
    sobel_y = sobel_x.transpose()

    binary_image_array = np.array(binary_image)

    foreground_mask = (binary_image_array == foreground) * 255

    edges_x = convolve(foreground_mask, sobel_x)
    edges_y = convolve(foreground_mask, sobel_y)
    edges = np.sqrt(edges_x**2 + edges_y**2)
    edges = (edges / edges.max()) * 255 # normalize to pixel range

    height, width = edges.shape
    edge_coordinates = [(y, x) for x in range(width) for y in range(height)
                        if edges[y, x] >= BINARIZE_THRESHOLD]

    return edge_coordinates


class ContourImage():
    def __init__(self, image: Image):
        self.image = image
        self.binarized_image = None

    def binarize(self) -> None:
        """
        Convert the image to a binary image.
        """
        grayscale_image = self.image.convert("L")
        grayscale_array = np.array(grayscale_image)
        binarized_array = (grayscale_array >= BINARIZE_THRESHOLD) * 1
        self.binarized_image = Image.fromarray(binarized_array.astype(np.uint8))

    def show(self) -> None:
        self.to_PIL().show()

    def fill_border(self):
        """
        Fill the border of the binarized image with zeros.
        """
        image_array = np.array(self.binarized_image)
        image_array = np.pad(image_array[1:-1, 1:-1], pad_width=1, mode="constant", constant_values=0)
        self.binarized_image = Image.fromarray(image_array.astype(np.uint8))

    def to_PIL(self) -> Image:
        color_array = np.stack([self.binarized_image]*3, axis=-1) * 255
        color_array = color_array.astype(np.uint8)
        return Image.fromarray(color_array)

    def prepare(self) -> np.ndarray:
        self.binarize()
        self.fill_border()
        return self.binarized_image


def find_chessboard_contours(image: Image) -> np.ndarray:
    image = ContourImage(image)
    return find_contours(image.prepare())

def draw_corners(pil_img: Image,
                 corners: np.ndarray,
                 color: tuple=(255, 0, 0),
                 radius: int=5) -> Image:
    img_with_corners = pil_img.copy()
    draw = ImageDraw.Draw(img_with_corners)

    for (y, x) in corners:
        left_up_point = (x - radius, y - radius)
        right_down_point = (x + radius, y + radius)
        draw.ellipse([left_up_point, right_down_point], outline=color, width=2)

    return img_with_corners

if __name__ == "__main__":
    if not os.path.exists(env.p1.output):
        os.makedirs(env.p1.output)
    # engine.get_distorted_chessboard(env.p1.chessboard_path)

    image = Image.open(env.p1.chessboard_path)
    contours = find_chessboard_contours(image)

    result_img = draw_corners(image, contours, color=(255, 0, 0), radius=5)
    result_img.save(env.p1.contours_path)
    plt.imshow(result_img)
    plt.title("Chessboard Contours")
    plt.show()