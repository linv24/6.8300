from pathlib import Path
from typing import Literal, TypedDict

from jaxtyping import Float
from torch import Tensor

import json
from PIL import Image
import torch
import torchvision


class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def get_kerberos() -> str:
    """Please return your kerberos ID as a string.
    This is required to match you with your specific puzzle dataset.
    """
    return "linv"


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""

    with open(f"{path}/metadata.json", "r") as f:
        metadata = json.load(f)
    extrinsics, intrinsics = torch.tensor(metadata["extrinsics"]), torch.tensor(metadata["intrinsics"])

    images_tensor = torch.empty((32, 256, 256))
    for image_ix in range(32):
        im = Image.open(f"{path}/images/{image_ix:02}.png")
        im_tensor = torchvision.transforms.functional.pil_to_tensor(im)[0]
        images_tensor[image_ix] = im_tensor

    return PuzzleDataset(
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        images=images_tensor,
    )


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    """

    transformation_matrix = torch.tensor([
        [ 0., -0., -1.],
        [ 0., -1., -0.],
        [ 1., -0., -0.]
    ])
    extrinsics = dataset["extrinsics"]
    R = extrinsics[..., :3, :3]
    transformed_R = R @ transformation_matrix
    extrinsics[..., :3, :3] = transformed_R
    dataset["extrinsics"] = extrinsics

    return dataset


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""

    return "c2w"


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""

    return "+x"


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""

    return "+y"


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""

    return "-z"


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    ret = """
    To determine the transformation matrix to convert to opencv format, I enumerated through
    (1) all permutations of the identity matrix, to find the orientation of the right, up, and
    look vectors, and (2) the sign of the columns of the transformation, to find the direction
    of each axis.
    I applied the transformation matris to the rotation matrix of the original extrinsics
    matrices (first 3 rows and columns) and reconstructed the transformed matrix
    to try to reconstruct the given images.
    I tested each transformation matrix on the original set of vertices to try to
    match the first example image in the dataset, and the transformation matrix used in
    convert_dataset reproduced the given dataset images.
    """
