import csv
import os

import numpy as np
from napari.utils.notifications import show_info
from scipy.optimize import fsolve
import pandas as pd
from scipy.optimize import minimize


def compute_center_of_mass(points):
    """Calcul du centre de masse."""
    center_of_mass = np.mean(points, axis=0)
    print(f"Centre de masse calculé : {center_of_mass}")
    return center_of_mass


def normalize_points(points, center_of_mass):
    """Centre et normalise les points par dimension."""
    centered_points = points - center_of_mass
    cov_matrix = np.cov(centered_points.T)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    scale_factors = np.sqrt(np.maximum(eigvals, 1e-8))
    P = eigvecs @ np.diag(1 / scale_factors)

    print(f"Matrice P :\n{P}")
    print(f"Produit P.T @ P :\n{P.T @ P}")

    normalized_points = (P @ centered_points.T).T
    print(f"Matrice de covariance : {cov_matrix}")
    print(f"Valeurs propres (normalisation) : {eigvals}")
    print(f"Matrice de transformation P :\n{P}")
    print(f"Points normalisés :\n{normalized_points}")
    print(f"Étendue des points normalisés : {np.ptp(normalized_points, axis=0)}")
    return normalized_points, P


def compute_k_matrix(points):
    """Calcul de la matrice K basée sur les points normalisés."""
    K = np.zeros((10, 10))
    for p in points:
        q = np.array([
            p[0]**2, p[1]**2, p[2]**2,
            np.sqrt(2) * p[0] * p[1], np.sqrt(2) * p[0] * p[2], np.sqrt(2) * p[1] * p[2],
            np.sqrt(2) * p[0], np.sqrt(2) * p[1], np.sqrt(2) * p[2], 1.0
        ])
        K += np.outer(q, q)
    print(f"Matrice K :\n{K}")
    return K


def initialize_q0():
    """Initialisation de q0."""
    q0 = np.zeros(10)
    q0[:3] = 1.0 / 3.0
    q0[9] = -1.0
    print(f"Initialisation de q0 : {q0}")
    return q0


def run_douglas_rachford(K, gamma=10, max_iter=10000, tol=1e-8):
    """Optimisation avec Douglas-Rachford."""
    M = gamma * K + np.eye(10)
    q = initialize_q0()
    p = q.copy()

    for iteration in range(max_iter):
        q = project_to_positive_semi_definite_with_trace(p)
        update = np.linalg.solve(M, 2 * q - p) - q
        p += update
        norm_update = np.linalg.norm(update)
        if norm_update < tol:
            print(f"Convergence atteinte à l'itération {iteration}")
            break
        if iteration % 100 == 0:
            print(f"Itération {iteration} - Norme de mise à jour : {norm_update}")
    print(f"Solution finale q : {q}")
    return q


def project_to_positive_semi_definite_with_trace(p):
    """Projection sur les matrices semi-définies positives avec contrainte de trace."""
    # Reconstruire la matrice A à partir de p
    A = np.array([
        [p[0], p[3] / np.sqrt(2), p[4] / np.sqrt(2)],
        [p[3] / np.sqrt(2), p[1], p[5] / np.sqrt(2)],
        [p[4] / np.sqrt(2), p[5] / np.sqrt(2), p[2]]
    ])

    # Décomposition propre de A
    eigvals, eigvecs = np.linalg.eigh(A)

    # Projection des valeurs propres sur les positives
    eigvals = np.maximum(eigvals, 0)

    print(f"Somme des valeurs propres projetées : {np.sum(eigvals)}")

    # Projection des valeurs propres sur le simplex
    eigvals = project_onto_simplex(eigvals)

    print(f"Somme des valeurs propres projetées : {np.sum(eigvals)}")

    # Reconstruire la matrice projetée A
    A_projected = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Reconstruire le vecteur projeté p
    projected_p = np.zeros(10)
    projected_p[0] = A_projected[0, 0]
    projected_p[1] = A_projected[1, 1]
    projected_p[2] = A_projected[2, 2]
    projected_p[3] = np.sqrt(2) * A_projected[0, 1]
    projected_p[4] = np.sqrt(2) * A_projected[0, 2]
    projected_p[5] = np.sqrt(2) * A_projected[1, 2]
    projected_p[6:] = p[6:]

    return projected_p


def project_onto_simplex(y):
    """Projette un vecteur y sur le simplex de somme égale à 1."""
    m = len(y)
    sorted_y = np.sort(y)[::-1]  # Tri décroissant
    tmpsum = 0

    for i in range(m - 1):
        tmpsum += sorted_y[i]
        tmax = (tmpsum - 1) / (i + 1)
        if tmax >= sorted_y[i + 1]:
            break
    else:
        tmax = (tmpsum + sorted_y[-1] - 1) / m

    return np.maximum(y - tmax, 0)


def extract_parameters(q):
    """Extrait A, b, et c à partir de q."""
    A = np.array([
        [q[0], q[3] / np.sqrt(2), q[4] / np.sqrt(2)],
        [q[3] / np.sqrt(2), q[1], q[5] / np.sqrt(2)],
        [q[4] / np.sqrt(2), q[5] / np.sqrt(2), q[2]]
    ])

    eigvals, _ = np.linalg.eigh(A)
    lengths_unnormalized = np.sqrt(1 / eigvals)
    print(f"Longueurs non normalisées des axes : {lengths_unnormalized}")

    b = np.array([q[6] / np.sqrt(2), q[7] / np.sqrt(2), q[8] / np.sqrt(2)])
    c = q[9]
    print(f"Paramètres extraits - Matrice A :\n{A}")
    print(f"Vecteur b : {b}")
    print(f"Scalaire c : {c}")
    return A, b, c


def denormalize_parameters(A, b, c, center_of_mass, P):
    """Dénormalise A, b et c."""
    A_denorm = P.T @ A @ P
    b_denorm = P.T @ b - 2 * A_denorm @ center_of_mass
    c_denorm = c + center_of_mass.T @ A_denorm @ center_of_mass - b.T @ P @ center_of_mass

    eigvals_denorm, eigvecs_denorm = np.linalg.eigh(A_denorm)
    print(f"Valeurs propres dénormalisées : {eigvals_denorm}")
    print(f"Longueurs des axes après dénormalisation : {np.sqrt(1 / eigvals_denorm)}")

    print("Matrice A dénormalisée :", A_denorm)
    print("Vecteur b dénormalisé :", b_denorm)
    print("Scalaire c dénormalisé :", c_denorm)
    return A_denorm, b_denorm, c_denorm


def calculate_centroid(A, b):
    """Calcul du centroide."""
    centroid = -np.linalg.solve(A, b)
    print(f"Centroide calculé : {centroid}")
    return centroid


def refine_scale_factor(A, centroid, eigvecs, points):
    """
    Calcule le facteur d'échelle optimal pour ajuster l'ellipsoide aux points.

    Args:
    - A: Matrice de l'ellipsoide.
    - centroid: Coordonnées du centroide.
    - eigvecs: Vecteurs propres de la matrice A.
    - points: Points d'entrée.

    Returns:
    - scale_factor: Facteur d'échelle ajusté.
    """
    eigvals, _ = np.linalg.eigh(A)
    axes_lengths = np.sqrt(1 / eigvals)  # Longueurs initiales des axes

    # Fonction pour calculer la distance d'un point à l'ellipsoide
    def distance_to_ellipsoid_scale(scale):
        scaled_axes = axes_lengths * scale
        distances = []
        for point in points:
            relative_point = eigvecs.T @ (point - centroid)  # Transformer le point dans le repère principal
            normalized_point = relative_point / scaled_axes  # Normaliser par les axes
            distance = np.linalg.norm(normalized_point) - 1  # Distance à la surface de l'ellipsoide
            distances.append(distance ** 2)
        return np.mean(distances)

    # Trouver le facteur d'échelle optimal
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(distance_to_ellipsoid_scale, bounds=(0.1, 10), method='bounded')
    scale_factor = result.x
    print(f"Facteur d'échelle optimisé : {scale_factor}")
    return scale_factor


def visualize_ellipsoid(A, centroid, eigvecs, viewer):
    """Visualisation de l'ellipsoide dans Napari avec rendu lisse."""
    # Calculer le facteur d'échelle optimal
    scale_factor = refine_scale_factor(A, centroid, eigvecs, viewer.layers["Points"].data)
    eigvals, _ = np.linalg.eigh(A)
    axes_lengths = np.sqrt(1 / eigvals) * scale_factor  # Appliquer le facteur d'échelle

    # Créer une grille dense pour une meilleure interpolation
    u = np.linspace(0, 2 * np.pi, 200)  # Augmenter la résolution ici
    v = np.linspace(0, np.pi, 100)
    u, v = np.meshgrid(u, v)

    # Points sur une sphère transformée en ellipsoide
    x = axes_lengths[0] * np.cos(u) * np.sin(v)
    y = axes_lengths[1] * np.sin(u) * np.sin(v)
    z = axes_lengths[2] * np.cos(v)

    # Appliquer les transformations pour les axes propres
    ellipsoid_points = eigvecs @ np.array([x.ravel(), y.ravel(), z.ravel()])
    x_rot, y_rot, z_rot = ellipsoid_points.reshape(3, x.shape[0], x.shape[1]) + centroid[:, None, None]

    # Conversion en surface
    vertices = np.column_stack((x_rot.ravel(), y_rot.ravel(), z_rot.ravel()))
    faces = []

    for i in range(u.shape[0] - 1):
        for j in range(u.shape[1] - 1):
            p1 = i * u.shape[1] + j
            p2 = p1 + u.shape[1]
            p3 = p1 + 1
            p4 = p2 + 1
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])

    faces = np.array(faces)

    # Ajouter la surface dans le viewer avec un rendu lisse et translucide pour voir les points et l'échantillon
    viewer.add_surface((vertices, faces), shading="smooth", name="Ellipsoide", opacity=0.10, blending="additive",
                       colormap="blue")


def orienter_axes(eigvecs, PA, PD, origine):
    """
    Fixe l'orientation de la base {u, v, w} de façon constante.

    - eigvecs : matrice 3x3 des vecteurs propres (on suppose qu'après tri, eigvecs[:, 0] correspond au grand axe).
    - PA      : point antérieur (vaut un vecteur de dimension 3)
    - PD      : point dorsal (vecteur de dimension 3)
    - origine : origine du repère (vecteur de dimension 3)

    Retourne :
      u, v, w : repère orthonormé avec v orienté vers PA et w vers PD.
    """
    PA = np.asarray(PA).ravel()
    PD = np.asarray(PD).ravel()
    origine = np.asarray(origine).ravel()

    # v est associé au grand axe (supposé être eigvecs[:,0] après tri)
    v = eigvecs[:, 0].copy()
    if np.dot(v, PA - origine) < 0:
        v = -v

    # Parmi les deux autres vecteurs, choisissez celui qui pointe le mieux vers PD
    cand1 = eigvecs[:, 1].copy()
    cand2 = eigvecs[:, 2].copy()
    d1 = np.dot(cand1, PD - origine)
    d2 = np.dot(cand2, PD - origine)
    if d1 >= d2:
        w = cand1 if d1 > 0 else -cand1
    else:
        w = cand2 if d2 > 0 else -cand2

    # Déduire u pour obtenir un repère droit
    u = np.cross(w, v)
    u /= np.linalg.norm(u)
    # Recalculer w pour assurer l'orthonormalité
    """w = np.cross(u, v)
    w /= np.linalg.norm(w)
    v /= np.linalg.norm(v)"""

    return u, v, w


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


def visualize_ellipsoid(A, centroid, eigvecs, viewer):
    """Visualisation de l'ellipsoide dans Napari avec rendu lisse."""
    # Calculer le facteur d'échelle optimal
    scale_factor = refine_scale_factor(A, centroid, eigvecs, viewer.layers["Points"].data)
    eigvals, _ = np.linalg.eigh(A)
    axes_lengths = np.sqrt(1 / eigvals) * scale_factor  # Appliquer le facteur d'échelle

    # Créer une grille dense pour une meilleure interpolation
    u = np.linspace(0, 2 * np.pi, 200)  # Augmenter la résolution ici
    v = np.linspace(0, np.pi, 100)
    u, v = np.meshgrid(u, v)

    # Points sur une sphère transformée en ellipsoide
    x = axes_lengths[0] * np.cos(u) * np.sin(v)
    y = axes_lengths[1] * np.sin(u) * np.sin(v)
    z = axes_lengths[2] * np.cos(v)

    # Appliquer les transformations pour les axes propres
    ellipsoid_points = eigvecs @ np.array([x.ravel(), y.ravel(), z.ravel()])
    x_rot, y_rot, z_rot = ellipsoid_points.reshape(3, x.shape[0], x.shape[1]) + centroid[:, None, None]

    # Conversion en surface
    vertices = np.column_stack((x_rot.ravel(), y_rot.ravel(), z_rot.ravel()))
    faces = []

    for i in range(u.shape[0] - 1):
        for j in range(u.shape[1] - 1):
            p1 = i * u.shape[1] + j
            p2 = p1 + u.shape[1]
            p3 = p1 + 1
            p4 = p2 + 1
            faces.append([p1, p2, p3])
            faces.append([p2, p4, p3])

    faces = np.array(faces)

    # Ajouter la surface dans le viewer avec un rendu lisse et translucide pour voir les points et l'échantillon
    viewer.add_surface((vertices, faces), shading="smooth", name="Ellipsoide", opacity=0.10, blending="additive",
                       colormap="blue")
