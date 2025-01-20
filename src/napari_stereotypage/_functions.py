import csv
import os

import numpy as np
from napari.utils.notifications import show_info
from scipy.optimize import fsolve
import pandas as pd
from scipy.optimize import minimize


def ajustement_moindres_carres_contraint(sommet, points):
    points_centres = points - sommet
    U, S, Vt = np.linalg.svd(points_centres)
    normal = Vt[-1]
    return normal


def calculate_apex_dorsal_plane(apex, dorsal, points):
    """
    Computation of the plane passing through apex and dorsal points and as close to the other four points as possible.
    :param apex: the apex point coordinates
    :param dorsal: the dorsal point coordinates
    :param points: the coordinates of the points defining the antero-posterior axis
    :return:
    """
    apex_dorsal_direction = dorsal - apex
    apex_dorsal_direction /= np.linalg.norm(apex_dorsal_direction)

    def initial_guess_svd(points, apex):
        points_centered = points - apex
        _, _, Vt = np.linalg.svd(points_centered)
        return Vt[-1]

    def objective_function(params):
        """
        Objective: Minimize the sum of squared distances of the additional points to the plane.
        """
        a, b, c = params
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)  # Normalize the normal vector
        d = -np.dot(normal, apex)  # Plane offset to pass through Apex

        # Ensure Apex and Dorsal are in the plane
        dist_apex = np.dot(apex, normal) + d
        dist_dorsal = np.dot(dorsal, normal) + d

        # Calculate distances of the other points to the plane
        distances = np.dot(points, normal) + d

        # Penalize deviations of apex and dorsal from the plane
        penalty = dist_apex**2 + dist_dorsal**2
        return np.sum(distances**2) + penalty


    def apex_dorsal_constraint(params):
        a, b, c = params
        normal = np.array([a, b, c])
        return np.dot(normal, apex_dorsal_direction)  # Should be 0 for orthogonality

    initial_guess = initial_guess_svd(points, apex)
    result = minimize(
        objective_function,
        initial_guess,
        method="SLSQP",
        constraints=[
            {"type": "eq", "fun": apex_dorsal_constraint},
        ],
        options={"maxiter": 2000},
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    normal = result.x / np.linalg.norm(result.x)
    d = -np.dot(normal, apex)
    return np.array([*normal, d])



def calculate_origin(apex, dorsal, normal, p, d):
    """
    Calculate the origin in the plane at a depth `p` from Dorsal
    and at a distance `d` from Apex.

    Args:
        apex (np.array): Coordinates of the Apex point (shape (3,)).
        dorsal (np.array): Coordinates of the Dorsal point (shape (3,)).
        normal (np.array): Normal vector of the plane (shape (3,)).
        p (float): Depth from Dorsal in the plane.
        d (float): Distance from Apex in the plane.

    Returns:
        np.array: Coordinates of the origin.
    """
    # Step 1: Define two vectors in the plane
    arbitrary_vector = np.array([1, 0, 0])
    if np.allclose(np.cross(normal, arbitrary_vector), 0):
        arbitrary_vector = np.array([0, 1, 0])

    v = np.cross(normal, arbitrary_vector)
    v = v / np.linalg.norm(v)  # Normalize
    w = np.cross(normal, v)
    w = w / np.linalg.norm(w)  # Normalize

    # Step 2: Define the function to solve
    def equations(params):
        alpha, beta = params
        # Origin in the plane
        origin = dorsal + alpha * v + beta * w

        # Constraints: Distance from Apex and depth from Dorsal
        eq1 = np.linalg.norm(origin - dorsal) - p
        eq2 = np.linalg.norm(origin - apex) - d
        return [eq1, eq2]

    # Step 3: Solve the equations
    initial_guess = [0, 0]
    alpha_beta = fsolve(equations, initial_guess)
    alpha, beta = alpha_beta

    # Step 4: Calculate the final origin
    origin = dorsal + alpha * v + beta * w
    return origin


def project_point_onto_plane(point, plane_point, normal):
    """
    Project a point onto a plane.

    Args:
        point (np.array): Coordinates of the point to project (shape (3,)).
        plane_point (np.array): A point on the plane (shape (3,)).
        normal (np.array): Normal vector of the plane (shape (3,)).

    Returns:
        np.array: Coordinates of the projected point (shape (3,)).
    """
    normal = normal / np.linalg.norm(normal)  # Ensure normal is unit length
    vector_to_plane = point - plane_point  # Vector from plane point to the point
    distance = np.dot(vector_to_plane, normal)  # Signed distance to the plane
    projection = point - distance * normal  # Projected point
    return projection


def calculate_frame_on_translated_line(dorsal, dorsal2_projected, apex, distance_apex, normal):
    """
    Calculate an orthonormal frame in the translated line passing through Apex.

    Args:
        dorsal (np.array): Coordinates of the Dorsal point (shape (3,)).
        dorsal2_projected (np.array): Coordinates of the projected Dorsal2 point (shape (3,)).
        apex (np.array): Coordinates of the Apex point (shape (3,)).
        distance_apex (float): Desired distance from Apex to the origin O (in micrometers).
        normal (np.array): Normal vector of the plane (shape (3,)).

    Returns:
        tuple: (origin, vector_x, vector_y, vector_z)
    """
    # Step 1: Calculate the direction vector of the line
    direction = dorsal2_projected - dorsal
    direction /= np.linalg.norm(direction)  # Normalize

    # Step 2: Calculate the origin O on the line
    t = distance_apex / np.linalg.norm(direction)
    o1 = apex - t * direction
    o2 = apex + t * direction
    distance_o1 = np.linalg.norm(o1 - dorsal)
    distance_o2 = np.linalg.norm(o2 - dorsal)
    origin = o1 if distance_o1 < distance_o2 else o2

    # Step 3: Calculate the y-vector (from origin to Apex)
    vector_y = apex - origin
    vector_y /= np.linalg.norm(vector_y)  # Normalize

    # Step 4: Calculate the z-vector to ensure it's oriented towards the dorsal plane
    # Compute a vector in the direction from the origin to dorsal
    dorsal_direction = dorsal - origin
    dorsal_direction /= np.linalg.norm(dorsal_direction)

    # Project `dorsal_direction` onto the plane orthogonal to vector_y
    vector_z = dorsal_direction - np.dot(dorsal_direction, vector_y) * vector_y
    vector_z /= np.linalg.norm(vector_z)  # Normalize

    # Step 5: Calculate the x-vector using y and z (ensuring orthogonality)
    vector_x = np.cross(vector_z, vector_y)
    vector_x /= np.linalg.norm(vector_x)  # Normalize

    return origin, vector_x, vector_y, vector_z



def calcul_repere(points, profondeur_um, distance_apex_um, dx, dy, dz):
    """
    Calculate an orthonormal reference frame in a plane containing Apex and Dorsal.
    The origin is placed at a given depth from Dorsal and at a given distance from Apex.

    Args:
        points (np.array): Array of points containing Apex, 4 points for plane fitting, and Dorsal.
                           Shape: (6, 3), where points[0] is Apex, points[-1] is Dorsal.
        profondeur_um (float): Depth from Dorsal for the origin (in micrometers).
        distance_apex_um (float): Distance from Apex for the origin (in micrometers).
        dx (float): Resolution of voxels in the x-direction (in micrometers/pixel).
        dy (float): Resolution of voxels in the y-direction (in micrometers/pixel).
        dz (float): Resolution of voxels in the z-direction (in micrometers/pixel).

    Returns:
        tuple: (centre_base, vecteur_u, vecteur_y, vecteur_w, sommet, vecteur_v)
    """
    # Step 1: Extract points
    sommet = points[0]  # Apex
    dorsal = points[-2]  # Dorsal
    dorsal2 = points[-1]  # Dorsal2 (last point to be projected onto the plane)
    points_plan = points[1:-2]  # The 4 points defining the plane

    # Step 2: Calculate the plane containing Apex and Dorsal
    plane_params = calculate_apex_dorsal_plane(sommet, dorsal, points_plan)
    normal = plane_params[:3]  # Normal vector (a, b, c)
    normal = normal / np.linalg.norm(normal)  # Normalize

    # Step 3: Project Dorsal2 onto the plane
    dorsal2_projected = project_point_onto_plane(dorsal2, sommet, normal)

    #Step 4 : Computation of the orthonormal frame on the dorsal-dorsal2 straight line translated to apex
    # Factor to convert distances to pixel space
    distance_apex_voxels = distance_apex_um / np.sqrt(dx**2 + dy**2 + dz**2)
    origin, vector_x, vector_y, vector_z = (
        calculate_frame_on_translated_line(dorsal, dorsal2_projected, sommet, distance_apex_voxels, normal))
    print("Distance Apex (µm):", distance_apex_um)
    print("Distance Apex (pixels):", distance_apex_voxels)
    calculated_distance_pixels = np.linalg.norm(origin - sommet)
    print("Calculated distance (pixels):", calculated_distance_pixels)

    return sommet, normal, dorsal2_projected, origin, vector_x, vector_y, vector_z


def ajuster_origine(centre_base, sommet, vecteur_y, hauteur_physique, dx, dy, dz):
    facteur = np.sqrt((vecteur_y[0] * dx) ** 2 + (vecteur_y[1] * dy) ** 2 + (vecteur_y[2] * dz) ** 2)
    hauteur_pixels = hauteur_physique / facteur
    vecteur_y_pixels = vecteur_y * hauteur_pixels / np.linalg.norm(vecteur_y)
    nouvelle_origine = sommet - vecteur_y_pixels
    return nouvelle_origine, hauteur_pixels


def projection_orthogonale_sur_plan(A, centre_base, v, w):
    n = np.cross(v, w)
    AC = A - centre_base
    projection_AC_sur_n = (np.dot(AC, n) / np.dot(n, n)) * n
    A_prime = A - projection_AC_sur_n
    return A_prime


def trouver_nouvelle_origine(S, P, vecteur_v, vecteur_w, d, z_mi_hauteur):
    def equations(vars):
        alpha, beta = vars
        O_prime = P + alpha * vecteur_v + beta * vecteur_w
        eq1 = np.linalg.norm(S - O_prime) - d
        eq2 = O_prime[0] - z_mi_hauteur
        return [eq1, eq2]

    alpha_beta_initial_guess = [1, 1]
    alpha_opt, beta_opt = fsolve(equations, alpha_beta_initial_guess)
    O_prime_opt = P + alpha_opt * vecteur_v + beta_opt * vecteur_w
    return O_prime_opt, alpha_opt, beta_opt


def ajuster_hauteur(centre_base, sommet, vecteur_u, vecteur_y, vecteur_w, points, hauteur_physique, dx, dy, dz):
    point_haut_proj = projection_orthogonale_sur_plan(points[5], centre_base, vecteur_y, vecteur_w)
    point_bas_proj = projection_orthogonale_sur_plan(points[6], centre_base, vecteur_y, vecteur_w)
    mi_hauteur = (point_haut_proj + point_bas_proj) / 2
    mi_hauteur_z = int(mi_hauteur[0])

    print("Expected height of the origin in Z:", mi_hauteur_z)

    nouvelle_origine, alpha, beta = trouver_nouvelle_origine(sommet, centre_base, vecteur_y, vecteur_w,
                                                             hauteur_physique, mi_hauteur_z)
    nouveau_vecteur_y = (sommet - nouvelle_origine) / np.linalg.norm(sommet - nouvelle_origine)
    nouveau_vecteur_w = np.cross(vecteur_u, nouveau_vecteur_y)
    nouveau_vecteur_w /= np.linalg.norm(nouveau_vecteur_w)

    return nouvelle_origine, nouveau_vecteur_y, nouveau_vecteur_w


def ajuster_origine_sur_droite_et_distance(S, P, V, centre_base, u, d):
    """
    Directly computes the origin on the line passing through the apex (S) and the midpoint of the dorsal-ventral
    points projected onto the base plane, ensuring the point is at a fixed distance (d) from the apex.

    Args:
    - S (array): Apex coordinates.
    - P (array): Dorsal point coordinates.
    - V (array): Ventral point coordinates.
    - centre_base (array): Center of the base of the cone.
    - u (array): Vector defining the base direction.
    - d (float): Desired distance from the apex.

    Returns:
    - O_prime (array): The recalculated origin coordinates.
    - v (array): Updated vertical direction vector.
    - w (array): Updated orthogonal vector.
    """
    # Project dorsal and ventral points onto the plane
    P_proj = projection_orthogonale_sur_plan(P, centre_base, u, np.cross(u, S - centre_base))
    V_proj = projection_orthogonale_sur_plan(V, centre_base, u, np.cross(u, S - centre_base))

    # Compute midpoint of the projected points
    M_proj = (P_proj + V_proj) / 2

    # Compute the direction vector of the line
    direction = M_proj - S
    norm_direction = np.linalg.norm(direction)
    direction /= norm_direction  # Normalize direction vector

    # Compute the scaling factor t for the desired distance
    #t = d / norm_direction

    # Compute the new origin
    O_prime = S + d * direction

    # Update the vertical and orthogonal vectors
    v = (S - O_prime) / np.linalg.norm(S - O_prime)
    w = np.cross(u, v)
    w /= np.linalg.norm(w)

    return O_prime, v, w, M_proj


def transformer_coordonnees(point, origine, u, v, w):
    transformation_matrix = np.array([u, v, w]).T
    translation = point - origine
    new_coords = np.linalg.solve(transformation_matrix, translation)
    return new_coords


def calculer_distance_physique(point_x, point_y, dx, dy, dz):
    differences = np.array([(point_y[0] - point_x[0]) * dx,
                            (point_y[1] - point_x[1]) * dy,
                            (point_y[2] - point_x[2]) * dz])
    distance_physique = np.linalg.norm(differences)
    return distance_physique


def save_data_to_csv(image_path, data):
    """
    Sauvegarde les données calculées dans un fichier CSV

    Args:
    - image_path (str): Chemin de l'image en cours de traitement.
    - data (tuple): Données contenant l'origine et les vecteurs u, v, w.
    """
    global csv_filename
    csv_filename = os.path.splitext(image_path)[0] + '.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Origine', 'u', 'v', 'w'])
        csv_writer.writerow(data)
    return csv_filename

def draw_plane(viewer, sommet, vecteur_v):
    """
    Dessine un plan en utilisant un quadrilatère approximant le plan calculé.

    Args:
    - sommet (np.array): Coordonnées du sommet du cône.
    - vecteur_v (np.array): Vecteur normal au plan.
    """
    scale_factor = 800  # Facteur d'échelle pour définir la taille du plan
    arbitrary_vector = np.array([1, 0, 0])
    if np.allclose(np.cross(arbitrary_vector, vecteur_v), 0):
        arbitrary_vector = np.array([0, 1, 0])

    in_plane_vector1 = np.cross(vecteur_v, arbitrary_vector)
    in_plane_vector1 = scale_factor * in_plane_vector1 / np.linalg.norm(in_plane_vector1)
    in_plane_vector2 = np.cross(vecteur_v, in_plane_vector1)
    in_plane_vector2 = scale_factor * in_plane_vector2 / np.linalg.norm(in_plane_vector2)

    # Calcul des coins du quadrilatère représentant le plan
    p1 = sommet + in_plane_vector1 + in_plane_vector2
    p2 = sommet + in_plane_vector1 - in_plane_vector2
    p3 = sommet - in_plane_vector1 - in_plane_vector2
    p4 = sommet - in_plane_vector1 + in_plane_vector2

    # Ajouter le quadrilatère au calque de formes avec remplissage
    viewer.add_shapes(data=[np.array([p1, p2, p3, p4])],
                      shape_type='polygon',
                      edge_width=1,
                      edge_color='cyan',
                      face_color='cyan',
                      opacity=0.3,
                      name='Plan')


# Fonction pour dessiner le cône de révolution
def draw_cone(viewer, sommet, centre_base, point_base, vecteur_u, vecteur_w):
    """
    Dessine une représentation simplifiée du cône de révolution.

    Args:
    - sommet (array): Coordonnées du sommet du cône.
    - centre_base (array): Coordonnées du centre de la base du cône.
    - point_base (array): Coordonnées d'un point sur la base.
    - vecteur_u (array): Vecteur directeur de la base.
    - vecteur_w (array): Vecteur perpendiculaire pour former le plan de la base.
    """
    # Calcul du rayon de la base à partir du point_base
    rayon_base = np.linalg.norm(point_base - centre_base)

    # Utilisation d'un nombre réduit de segments pour la base
    num_segments = 12  # Par exemple, 12 segments
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)

    # Points de la base
    base_points = [centre_base + rayon_base * (np.cos(angle) * vecteur_u + np.sin(angle) * vecteur_w) for angle in
                   angles]
    base_points = np.array(base_points)

    # Ajouter le polygone de la base
    viewer.add_shapes(data=[base_points], shape_type='polygon', edge_width=1, edge_color='yellow', face_color='yellow',
                      opacity=0.3, name='Base du cône')

    # Ajouter les lignes reliant le sommet à la base
    for point in base_points:
        viewer.add_shapes(data=np.array([sommet, point]), shape_type='line', edge_width=1, edge_color='yellow',
                          name='Arête du cône')


# Fonction pour dessiner le repère
def draw_axes(viewer, origine, u, v, w):
    """
    Dessine les 3 axes d'un repère

    Args:
    - origine (tuple): Coordonnées de l'origine du nouveau repère (x0, y0, z0).
    - vecteur_u (tuple): Coordonnées du vecteur unitaire u (ux, uy, uz).
    - vecteur_v (tuple): Coordonnées du vecteur unitaire v (vx, vy, vz).
    - vecteur_w (tuple): Coordonnées du vecteur unitaire w (wx, wy, wz).
    """
    # Dessine l'axe x
    segment_u = np.array([origine, origine + 50 * u])
    viewer.add_shapes(segment_u, shape_type='line', edge_width=1, edge_color='blue', name='Axe X')

    # Dessine l'axe y
    segment_v = np.array([origine, origine + 50 * v])
    viewer.add_shapes(segment_v, shape_type='line', edge_width=1, edge_color='red', name='Axe Y')

    # Dessine l'axe z
    segment_w = np.array([origine, origine + 50 * w])
    viewer.add_shapes(segment_w, shape_type='line', edge_width=1, edge_color='green', name='Axe Z')

    viewer.dims.ndisplay = 3


# Fonction pour dessiner l'origine du repère
def draw_origin(viewer, origine):
    """
    Dessine un point rouge à l'origine du repère.

    Args:
    - origine (tuple): Coordonnées de l'origine du repère (centre de la base du cône).
    """
    viewer.add_points([origine], face_color='red', border_color='red', size=5, name='Origine')
