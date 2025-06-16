"""
This script loads a 3D image aligned in a local coordinate system (e.g., fitted ellipsoid frame)
and reprojects it into the original image space using an affine transformation.

It:
- Loads a TIFF image and a CSV file containing the local coordinate frame: origin, and orthonormal axes (u, v, w).
- Constructs the affine matrix to map the local frame back into the global image reference.
- Recenters the local origin to a desired output center (default: [300, 300, 300]).
- Applies the inverse affine transform to produce the reprojected image in global space.
- Visualizes both the local and transformed images using Napari, along with the origin point.
"""


import numpy as np
from tifffile import imread
from scipy.ndimage import affine_transform
import napari
import re

# === 1. Parseur CSV ===
def parse_vector(s):
    """
    Parses a string representation of a vector formatted as "[x y z]".

    Parameters:
    - s (str): String containing a 3D vector with space-separated values and brackets.

    Returns:
    - ndarray: 1D NumPy array of floats.
    """
    s = s.strip().replace('[', '').replace(']', '')
    parts = s.split()
    return np.array([float(p) for p in parts])

def read_frame_csv(csv_path):
    """
    Parses a CSV file containing the definition of a local orthonormal frame
    (typically fitted from an ellipsoid) with origin and axes vectors.

    The CSV is expected to contain one line with 4 vectors written as:
    [origin], [u], [v], [w]
    Each vector should be formatted like "[x y z]" with space-separated values.

    Parameters:
    - csv_path (str): Path to the CSV file containing the frame parameters.

    Returns:
    - origin (ndarray): 3-element array with the coordinates of the origin in ZYX image order.
    - u (ndarray): 3-element array representing the X-axis of the local frame.
    - v (ndarray): 3-element array representing the Y-axis of the local frame.
    - w (ndarray): 3-element array representing the Z-axis of the local frame.
    """

    with open(csv_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    data_line = lines[1]  # ou lines[0] si pas d'entête

    # Utiliser une regex pour extraire tous les vecteurs entre crochets
    matches = re.findall(r'\[.*?\]', data_line)
    if len(matches) != 4:
        raise ValueError(f"Expected 4 vectors, got {len(matches)}")

    origin = parse_vector(matches[0])
    u = parse_vector(matches[1])
    v = parse_vector(matches[2])
    w = parse_vector(matches[3])
    return origin, u, v, w

# === 2. Transformation vers le repère image ===
def transform_image_to_global(image_local, R_zyx, origin_zyx, output_shape=None, new_center=(300, 300, 300), order=5):
    """
    Applies an affine transformation to reproject a 3D image from a local coordinate system
    (e.g., defined by a fitted ellipsoid) back into the global image coordinate space. This enables to register several
    samples onto a common reference frame.

    Parameters:
    - image_local (ndarray): 3D image defined in the local coordinate system.
    - R_zyx (ndarray): 3×3 matrix where columns are the orthonormal axes [w, v, u]
                       of the local frame, expressed in ZYX (image) order.
    - origin_zyx (ndarray): Coordinates (Z, Y, X) of the local frame origin in the original image space.
    - output_shape (tuple, optional): Shape of the output image. If None, defaults to the input shape.
    - new_center (tuple): Desired location (Z, Y, X) of the local origin in the output image.
                          Default is (300, 300, 300); using (0, 0, 0) may lead to clipping.
    - order (int): Interpolation order for the affine transform.
                   0 = nearest, 1 = linear, 3 = cubic, 5 = quintic (higher = better quality but slower).

    Returns:
    - image_global (ndarray): Transformed 3D image in the global coordinate system,
                              centered at `new_center`.
    """

    R_inv = np.linalg.inv(R_zyx)

    # Compute bounding box of transformed image
    z, y, x = image_local.shape
    corners = np.array([
        [0, 0, 0],
        [0, 0, x - 1],
        [0, y - 1, 0],
        [0, y - 1, x - 1],
        [z - 1, 0, 0],
        [z - 1, 0, x - 1],
        [z - 1, y - 1, 0],
        [z - 1, y - 1, x - 1],
    ])

    transformed_corners = corners @ R_inv.T - R_inv @ origin_zyx
    min_coord = np.floor(transformed_corners.min(axis=0)).astype(int)
    max_coord = np.ceil(transformed_corners.max(axis=0)).astype(int)
    output_shape_auto = tuple((max_coord - min_coord).tolist())

    if output_shape is None:
        output_shape = output_shape_auto

    offset = -R_inv @ origin_zyx - min_coord

    print("Offset:", offset)
    print("Output shape:", output_shape)
    print("Transformed Z range:", [min_coord[0], max_coord[0]])

    image_global = affine_transform(
        image_local,
        R_inv,
        offset=offset,
        output_shape=output_shape,
        order=order,
        mode='constant',
        cval=0.0
    )

    return image_global

# === 3. Main ===
image_path = "/home/clement/Images/Image_ellipse/20250131_cldnGFP_84nlsEOS_postPhotoconversion.lif - E5 - C=0.tif"
csv_path = "/home/clement/Images/Image_ellipse/20250131_cldnGFP_84nlsEOS_postPhotoconversion.lif - E5 - C=0.csv"

image_local = imread(image_path)
origin, u, v, w = read_frame_csv(csv_path)
R = np.stack([w, v, u], axis=1)
image_global = transform_image_to_global(image_local, R, origin)

# === 4. Visualisation Napari ===
viewer = napari.Viewer()
viewer.add_image(image_local, name='repère local')
viewer.add_image(image_global, name='repère image')
viewer.add_points([origin[::-1]], size=6, name='origine (global)', face_color='red')
napari.run()

