from jaxtyping import Float
from torch import Tensor
import torch


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""

    ones = torch.ones_like(points[..., :1])
    return torch.cat([points, ones], dim=-1)


def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""

    zeros = torch.zeros_like(points[..., :1])
    return torch.cat([points, zeros], dim=-1)


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""

    return transform @ xyz


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """

    world2cam = torch.inverse(cam2world)
    return world2cam @ xyz


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """

    return cam2world @ xyz


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""

    # intrinsics is a 3x3 matrix; need a 3x4 homogenized matrix
    homogenized_intrinsics = homogenize_vectors(intrinsics)
    homogenized_pixel_coordinates = xyz @ homogenized_intrinsics.transpose(-2, -1)
    w = homogenized_pixel_coordinates[..., -1].unsqueeze(-1)
    return (homogenized_pixel_coordinates / w)[..., :-1]
