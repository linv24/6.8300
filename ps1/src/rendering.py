from jaxtyping import Float
from torch import Tensor
import torch
import src.geometry as geometry

def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """

    vertex_size = vertices.shape[0]
    batch_size = extrinsics.shape[0]
    canvas_size = resolution[0]
    canvases = torch.full((batch_size, canvas_size, canvas_size), 255, dtype=torch.float32)

    homogenized_vertices = geometry.homogenize_points(vertices)
    cam_coordinates = homogenized_vertices.unsqueeze(0) @ torch.inverse(extrinsics).transpose(1, 2)
    # cam_coordinates: (batch, vertex, 4)
    # intrinsics: (batch, 3, 3)
    pixel_coordinates = geometry.project(cam_coordinates, intrinsics.unsqueeze(1))[0]

    for batch in range(batch_size):
        for vertex in range(vertex_size):
            coords = pixel_coordinates[batch, vertex] * canvas_size
            if ((coords >= 0) & (coords < canvas_size)).all():
                # coords in format (x, y), but need to paint canvas as [batch, y, x]
                canvases[batch, coords[1].long(), coords[0].long()] = 0

    return canvases