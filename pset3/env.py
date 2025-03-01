from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent
project_data = PROJECT_DIR / 'data'
src = PROJECT_DIR / 'src'
output = PROJECT_DIR / 'outputs'

class p1:
    data = project_data / 'p1_edge_identification'
    output = output / 'p1_edge_identification'
    chessboard_path = data / 'chessboard.png'
    contours_path = output / 'contours.png'

class p2:
    data = project_data / 'p2_calibrate_camera'
    output = output / 'p2_calibrate_camera'
    chessboard_corners = output / 'corners.png'
    camera_matrix = output / 'camera_matrix.npy'
    dist_coeff = output / 'dist_coeff.npy'
    undistorted_image = output / 'undistorted_image.png'

class p3:
    data = project_data / 'p3_fundamental_matrix'
    output = output / 'p3_fundamental_matrix'
    test_obj = data / 'test.obj'
    test_texture = data / 'test.mtl'
    im1 = output / 'im1.png'
    im2 = output / 'im2.png'
    pts_1 = data / 'pts_1.txt'
    pts_2 = data / 'pts_2.txt'

    const_im1 = data / 'const_im1.png'
    const_im2 = data / 'const_im2.png'

    lls_img = output / 'lls_img.png'
    norm_img = output / 'norm_img.png'


class p4:
    data = project_data / 'p4_image_rectification'
    output = output / 'p4_image_rectification'
    aligned_epipolar = output / 'aligned_epipolar.png'
    cv_matches = output / 'cv_matches.png'

class p5:
    data = project_data / 'p5_3D_reconstruction'
    output = output / 'p5_3D_reconstruction'
    arc_obj = data / 'arc_de_triomphe' / 'model.obj'
    arc_texture = data / 'arc_de_triomphe' / 'model.mtl'
    chessboard = data / 'chessboard.png'
    raw_images = data / 'raw_images'
    undistorted_images = output / 'undistorted_images'
    rotation_matrix = output / 'rotation_matrix.npy'
    translation_matrix = output / 'translation_matrix.npy'
    pointcloud = output / 'pointcloud.npy'

class p6:
    data = project_data / 'p6_SfM_pipeline'
    output = output / 'p6_SfM_pipeline'
    statue_images = data / 'statue'
    pointcloud = output / 'pointcloud.npy'