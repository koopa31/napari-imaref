import csv
import os

import numpy as np
from scipy.optimize import fsolve
import pandas as pd


def ajustement_moindres_carres_contraint(sommet, points):
    points_centres = points - sommet
    U, S, Vt = np.linalg.svd(points_centres)
    normal = Vt[-1]
    return normal


def calcul_repere(points):
    sommet = points[0]
    points_plan = points[1:5]
    point_haut = points[5]
    point_bas = points[6]
    point_base = points[7]

    vecteur_v = ajustement_moindres_carres_contraint(sommet, points_plan)
    vecteur_v = vecteur_v / np.linalg.norm(vecteur_v)

    distance_au_plan = np.dot(point_base - sommet, vecteur_v)
    point_sym = point_base - 2 * distance_au_plan * vecteur_v

    centre_base = (point_base + point_sym) / 2

    vecteur_y = sommet - centre_base
    vecteur_y = vecteur_y / np.linalg.norm(vecteur_y)

    vecteur_u = point_base - centre_base
    vecteur_u = vecteur_u / np.linalg.norm(vecteur_u)

    vecteur_w = np.cross(vecteur_u, vecteur_y)
    vecteur_w = vecteur_w / np.linalg.norm(vecteur_w)

    direction_bas = point_bas - centre_base
    if np.dot(vecteur_w, direction_bas) < 0:
        vecteur_w = -vecteur_w
        vecteur_u = np.cross(vecteur_y, vecteur_w)
        vecteur_u = vecteur_u / np.linalg.norm(vecteur_u)

    return centre_base, vecteur_u, vecteur_y, vecteur_w, sommet, vecteur_v, point_haut, point_bas


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
    scale_factor = 300  # Facteur d'échelle pour définir la taille du plan
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
