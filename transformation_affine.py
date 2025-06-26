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
from skimage.io import imsave, imread
from scipy.ndimage import affine_transform
import napari
from pathlib import Path
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

def compute_transformed_bbox(shape, R_inv, origin_zyx):
    z, y, x = shape
    corners = np.array([
        [0, 0, 0],
        [0, 0, x-1],
        [0, y-1, 0],
        [0, y-1, x-1],
        [z-1, 0, 0],
        [z-1, 0, x-1],
        [z-1, y-1, 0],
        [z-1, y-1, x-1],
    ])
    transformed = corners @ R_inv.T - R_inv @ origin_zyx
    min_coord = np.floor(transformed.min(axis=0)).astype(int)
    max_coord = np.ceil(transformed.max(axis=0)).astype(int)
    return min_coord, max_coord

# === 2. Transformation vers le repère image ===
def transform_image_to_global(image_local, R_zyx, origin_zyx, min_coord_ref, output_shape=None, order=5):
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

    if output_shape==None:
        output_shape_vec = np.asarray(image_local.shape)
    else:
        output_shape_vec = np.asarray(output_shape)
    #R_inv = np.linalg.inv(R_zyx)
    #offset = -R_inv @ origin_zyx - min_coord_ref
    offset = origin_zyx - R_zyx @ (output_shape_vec/2 - R_zyx.T @ (np.asarray(image_local.shape)/2- origin_zyx))
    # Explications de offset:
    # - origin_zyx décale l'image pour placer le centre de la rotation à l'origine du repère
    # - Le terme R_zyx @ (vecteur) correspond à un décalage après la transformation de vecteur
    # - si vecteur = output_shape_vec/2, on place le centre de rotation au centre de l'image
    # - avec dans vecteur le terme  - R_zyx.T @ (np.asarray(image_local)/2- origin_zyx), on translate
    # l'image du vecteur OC ou O est le centre de rotation et C et le centre de l'image après transformation
    # ce faisant, on place l'ancien centre de l'image au centre de la nouvelle image
    registered_image = affine_transform(
        image_local,
        R_zyx,
        offset=offset,
        output_shape=output_shape,
        order=order,
        mode='constant',
        cval=0.0
    )
    return registered_image

# === 3. Main ===
#image_path = "/home/clement/Images/Image_ellipse/20250131_cldnGFP_84nlsEOS_postPhotoconversion.lif - E5 - C=0.tif"
# image_path = "/home/clement/Images/caracterisation_16hpf/test/20250529_OmpYFP_84mKate_84nlsEOS_O40_02u_100u_16hpf_w1CSU-561 Em 593LP_s1.TIF"
image_path = "/Users/dsanchez/Downloads/20250529_OmpYFP_84mKate_84nlsEOS_O40_02u_100u_16hpf_w1CSU-561 Em 593LP_s1/20250529_OmpYFP_84mKate_84nlsEOS_O40_02u_100u_16hpf_w1CSU-561 Em 593LP_s1.TIF"
#csv_path = "/home/clement/Images/Image_ellipse/20250131_cldnGFP_84nlsEOS_postPhotoconversion.lif - E5 - C=0.csv"
# csv_path = "/home/clement/Images/caracterisation_16hpf/test/20250529_OmpYFP_84mKate_84nlsEOS_O40_02u_100u_16hpf_w1CSU-561 Em 593LP_s1.csv"
csv_path = "/Users/dsanchez/Downloads/20250529_OmpYFP_84mKate_84nlsEOS_O40_02u_100u_16hpf_w1CSU-561 Em 593LP_s1/20250529_OmpYFP_84mKate_84nlsEOS_O40_02u_100u_16hpf_w1CSU-561 Em 593LP_s1.csv"

image_local = imread(image_path)
origin, u, v, w = read_frame_csv(csv_path)
R = np.stack([w, v, u], axis=1)
#image_global = transform_image_to_global(image_local, R, origin)

R_inv_ref = R.T
min_coord_ref, max_coord_ref = compute_transformed_bbox(image_local.shape, R_inv_ref, origin)
output_shape = tuple((max_coord_ref - min_coord_ref).tolist())

image_global = transform_image_to_global(image_local, R, origin, min_coord_ref, output_shape, order=0)



# === 4. Visualisation Napari ===
viewer = napari.Viewer()
viewer.add_image(image_local, name='repère local')
viewer.add_image(image_global, name='repère image')
viewer.add_points([origin], size=6, name='origine (global)', face_color='red')

segment_u = np.array([origin, origin + 50 * u])
viewer.add_shapes(segment_u, shape_type='line', edge_width=1, edge_color='blue', name='Axe X')

# Dessine l'axe y
segment_v = np.array([origin, origin + 50 * v])
viewer.add_shapes(segment_v, shape_type='line', edge_width=1, edge_color='red', name='Axe Y')

# Dessine l'axe z
segment_w = np.array([origin, origin + 50 * w])
viewer.add_shapes(segment_w, shape_type='line', edge_width=1, edge_color='green', name='Axe Z')

viewer.dims.ndisplay = 3
napari.run()


"""
# === Traitement de tout un dossier ===
input_dir = Path("/home/clement/Images/caracterisation_16hpf")
images = sorted([
    f for ext in ("*.tif", "*.TIF")
    for f in input_dir.glob(ext)
    if "_cp_masks" not in f.stem
])
csvs = sorted(input_dir.glob("*.csv"))

# Affichage de debug
print("Images trouvées :", [img.name for img in images])
print("CSVs trouvés :", [csv.name for csv in csvs])

# Associer chaque image avec son masque et son CSV
image_csv_mask_triplets = []
for img in images:
    if "_cp_masks_clean" in img.stem:
        continue
    base_name = img.stem
    mask_path = input_dir / f"{base_name}_cp_masks_clean.tif"
    csv_path = input_dir / f"{base_name}.csv"
    if mask_path.exists() and csv_path.exists():
        image_csv_mask_triplets.append((img, mask_path, csv_path))
    else:
        print(f"[WARNING] Fichiers manquants pour {base_name}:")
        if not mask_path.exists():
            print("  - Masque manquant :", mask_path.name)
        if not csv_path.exists():
            print("  - CSV manquant :", csv_path.name)

print(f"Nombre de triplets valides trouvés : {len(image_csv_mask_triplets)}")

if image_csv_mask_triplets:
    # Utiliser la première image comme référence pour le min_coord commun
    ref_img_path, ref_mask_path, ref_csv_path = image_csv_mask_triplets[0]
    ref_image = imread(ref_img_path)
    ref_origin, ref_u, ref_v, ref_w = read_frame_csv(ref_csv_path)
    R_ref = np.stack([ref_w, ref_v, ref_u], axis=1)
    R_inv_ref = np.linalg.inv(R_ref)
    min_coord_ref, max_coord_ref = compute_transformed_bbox(ref_image.shape, R_inv_ref, ref_origin)
    output_shape = tuple((max_coord_ref - min_coord_ref).tolist())

    # Dossier de sortie
    output_dir = input_dir / "registered"
    output_dir.mkdir(exist_ok=True)

    # Traitement de tous les fichiers
    for img_path, mask_path, csv_path in image_csv_mask_triplets:
        image = imread(img_path)
        mask = imread(mask_path)
        origin, u, v, w = read_frame_csv(csv_path)
        R = np.stack([w, v, u], axis=1)

        image_registered = transform_image_to_global(image, R, origin, min_coord_ref, output_shape, order=0)
        mask_registered = transform_image_to_global(mask, R, origin, min_coord_ref, output_shape, order=0)

        imsave(output_dir / f"{img_path.stem}_registered.tif", image_registered.astype(np.uint16))
        imsave(output_dir / f"{mask_path.stem}_registered.tif", mask_registered.astype(np.uint16))

    print(f"✅ {len(image_csv_mask_triplets)} images et masques enregistrés dans : {output_dir}")
else:
    print("❌ Aucun triplet image/masque/csv valide trouvé.")

"""

