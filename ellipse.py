import numpy as np
import napari
from skimage.io import imread
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from joblib import Parallel, delayed


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


def run_douglas_rachford(K, gamma=10, max_iter=1000, tol=1e-8):
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
    """Calcul du centroïde."""
    centroid = -np.linalg.solve(A, b)
    print(f"Centroïde calculé : {centroid}")
    return centroid


def refine_scale_factor(A, centroid, eigvecs, points):
    """
    Calcule le facteur d'échelle optimal pour ajuster l'ellipsoïde aux points.

    Args:
    - A: Matrice de l'ellipsoïde.
    - centroid: Coordonnées du centroïde.
    - eigvecs: Vecteurs propres de la matrice A.
    - points: Points d'entrée.

    Returns:
    - scale_factor: Facteur d'échelle ajusté.
    """
    eigvals, _ = np.linalg.eigh(A)
    axes_lengths = np.sqrt(1 / eigvals)  # Longueurs initiales des axes

    # Fonction pour calculer la distance d'un point à l'ellipsoïde
    def distance_to_ellipsoid_scale(scale):
        scaled_axes = axes_lengths * scale
        distances = []
        for point in points:
            relative_point = eigvecs.T @ (point - centroid)  # Transformer le point dans le repère principal
            normalized_point = relative_point / scaled_axes  # Normaliser par les axes
            distance = np.linalg.norm(normalized_point) - 1  # Distance à la surface de l'ellipsoïde
            distances.append(distance ** 2)
        return np.mean(distances)

    # Trouver le facteur d'échelle optimal
    from scipy.optimize import minimize_scalar
    result = minimize_scalar(distance_to_ellipsoid_scale, bounds=(0.1, 10), method='bounded')
    scale_factor = result.x
    print(f"Facteur d'échelle optimisé : {scale_factor}")
    return scale_factor


def visualize_ellipsoid(A, centroid, eigvecs, viewer):
    """Visualisation de l'ellipsoïde dans Napari avec rendu lisse."""
    # Calculer le facteur d'échelle optimal
    scale_factor = refine_scale_factor(A, centroid, eigvecs, viewer.layers["Points"].data)
    eigvals, _ = np.linalg.eigh(A)
    axes_lengths = np.sqrt(1 / eigvals) * scale_factor  # Appliquer le facteur d'échelle

    # Créer une grille dense pour une meilleure interpolation
    u = np.linspace(0, 2 * np.pi, 200)  # Augmenter la résolution ici
    v = np.linspace(0, np.pi, 100)
    u, v = np.meshgrid(u, v)

    # Points sur une sphère transformée en ellipsoïde
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
    viewer.add_surface((vertices, faces), shading="smooth", name="Ellipsoïde", opacity=0.10, blending="additive",
                       colormap="blue")


def draw_axes(viewer, origine, u, v, w, std_label=""):
    """
    Dessine les 3 axes d'un repère et ajoute un label pour distinguer les niveaux de bruit.

    Args:
      - origine (array-like): Coordonnées de l'origine du repère (x0, y0, z0).
      - u (array-like): Vecteur unitaire pour l'axe X.
      - v (array-like): Vecteur unitaire pour l'axe Y.
      - w (array-like): Vecteur unitaire pour l'axe Z.
      - std_label (str, optionnel): Chaîne de caractères à ajouter au nom des axes.
    """
    # Axe X en bleu
    segment_u = np.array([origine, origine + 10 * u])
    viewer.add_shapes(segment_u, shape_type='line', edge_width=1, edge_color='blue',
                      name=f'Axe X {std_label}')
    # Axe Y en rouge
    segment_v = np.array([origine, origine + 10 * v])
    viewer.add_shapes(segment_v, shape_type='line', edge_width=1, edge_color='red',
                      name=f'Axe Y {std_label}')
    # Axe Z en vert
    segment_w = np.array([origine, origine + 10 * w])
    viewer.add_shapes(segment_w, shape_type='line', edge_width=1, edge_color='green',
                      name=f'Axe Z {std_label}')
    viewer.dims.ndisplay = 3


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


### SENSITIVITY ANALYSIS ###


def compute_orientation_deviation(original_axes, perturbed_axes):
    """
    Calcule la déviation angulaire des axes principaux entre la configuration originale et perturbée,
    en s'assurant que les axes gardent une orientation cohérente.

    Args:
        original_axes (np.ndarray): Matrice (3x3) des vecteurs propres des axes originaux.
        perturbed_axes (np.ndarray): Matrice (3x3) des vecteurs propres des axes perturbés.

    Returns:
        np.ndarray: Matrice (1x3) des déviations angulaires (en degrés) des trois axes.
    """
    deviations = []

    for i in range(3):  # Pour chaque axe (X, Y, Z)
        # Vérifier si l'axe perturbé est dans la même direction que l'axe original
        if np.dot(original_axes[:, i], perturbed_axes[:, i]) < 0:
            perturbed_axes[:, i] *= -1  # Inverser le signe si nécessaire

        # Calcul de l'angle entre les axes
        dot_product = np.clip(np.dot(original_axes[:, i], perturbed_axes[:, i]), -1.0, 1.0)
        angle = np.arccos(dot_product) * (180 / np.pi)  # Convertir en degrés
        deviations.append(angle)

    return np.array(deviations)

def plot_orientation_deviation(orientation_deviation):
    """
    Affiche un graphe de la déviation angulaire des axes.

    Args:
        orientation_deviation (np.ndarray): Matrice (Nx3) des déviations angulaires en degrés.
    """
    plt.figure(figsize=(10, 5))

    orientation_deviation = np.array(orientation_deviation).squeeze()

    for i, label, color in zip(range(3), ["Axe X", "Axe Y", "Axe Z"], ["red", "blue", "green"]):
        plt.plot(orientation_deviation[:, i], label=label, color=color)

    plt.xlabel("Simulation")
    plt.ylabel("Déviation angulaire (degrés)")
    plt.title("Sensibilité des orientations des axes")
    plt.legend()
    plt.grid()
    plt.show()


import numpy as np

def perturb_points_anisotropic(points, noise_std, voxel_size, isotropic):
    """
    Simule l'erreur de positionnement en tenant compte de l'anisotropie de résolution.
    L'erreur est appliquée directement en unités de voxels.

    Args:
        points (np.ndarray): Points originaux (N,3) exprimés en voxels.
        noise_std_voxel (float): Écart-type du bruit en unités de voxels.
        voxel_size (tuple): Tailles des voxels (dz, dy, dx).

    Returns:
        np.ndarray: Points perturbés (N,3) en voxels.
    """
    from scipy.stats import truncnorm
    if isotropic:

        """# 3D version
        n_points = points.shape[0]

        # Générer les rayons avec une loi normale tronquée entre -max_radius et +max_radius
        std = 1
        max_radius = noise_std * std
        r = truncnorm.rvs(-max_radius / std, max_radius / std, loc=0, scale=std, size=n_points)

        # Générer des directions aléatoires uniformes sur la sphère
        u = np.random.normal(size=(n_points, 3))  # Vecteurs aléatoires
        u /= np.linalg.norm(u, axis=1, keepdims=True)  # Normalisation (unitaire)

        # Appliquer le déplacement
        perturbations = u * r[:, np.newaxis]

        # Retourner les points perturbés
        return points + perturbations"""

        n_points = points.shape[0]

        # Définir le rayon maximal L = noise_std
        L = noise_std

        # Définir l'écart-type de la normale tronquée
        ratio_std = 0.2
        sigma = ratio_std * L  # Ajustez ratio_std selon la pente de décroissance voulue

        # Paramètres pour la loi normale tronquée (mu=0, sigma, intervalle [-L, L])
        mu = 0
        a, b = -L / sigma, L / sigma  # bornes en unités de l'écart-type

        # Générer des échantillons r dans [-L, +L]
        r = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_points)

        """
        # Affichage de l'histogramme de r
        plt.figure(figsize=(8, 5))
        plt.hist(r, bins=30, density=True, alpha=0.6, color='gray', edgecolor='black',
                 label="Histogramme des échantillons")

        # Tracer la densité théorique de la loi normale tronquée
        x = np.linspace(-L, L, 200)
        pdf = truncnorm.pdf(x, a, b, loc=mu, scale=sigma)
        plt.plot(x, pdf, 'b-', label=f"TruncNormal(mu=0, sigma={sigma:.2f}, interval=[-L,L])")

        plt.xlabel("r")
        plt.ylabel("Densité de probabilité")
        plt.legend()
        plt.title(f"Loi normale tronquée dans [-{L:.2f}, +{L:.2f}], sigma={sigma:.2f}")
        plt.show()"""

        # Générer des directions aléatoires uniformes sur un cercle
        angles = np.random.uniform(0, 2 * np.pi, size=n_points)
        dx = r * np.cos(angles)
        dy = r * np.sin(angles)

        # Appliquer le déplacement uniquement en XY
        perturbed_points = points.copy()
        perturbed_points[:, 1] += dx
        perturbed_points[:, 2] += dy

    else:
        # Normalisation de l'erreur en fonction de la résolution relative des voxels
        # Plus dz est grand, plus on réduit le bruit en Z en pixels.
        scaling_factor = np.array(voxel_size) / np.max(voxel_size)

        # Générer un bruit gaussien isotrope en voxels
        noise_voxels = np.random.normal(0, noise_std, points.shape)

        # Appliquer la correction d'anisotropie
        noise_voxels *= scaling_factor  # Réduit le bruit en pixels là où la résolution est plus faible

        # Ajouter le bruit en unités de voxels
        perturbed_points = points + noise_voxels

    return perturbed_points


def test_sensitivity(points, n_simulations=50, noise_std=1.0, isotropic=True):
    """
    Test de sensibilité au placement des points.

    Args:
        points (np.ndarray): Points originaux.
        n_simulations (int): Nombre de simulations avec perturbations.
        noise_std (float): Écart-type du bruit gaussien ajouté aux points.

    Returns:
        dict: Résultats des centroïdes, axes principaux et orientations pour chaque simulation.
    """
    results = {"centroids": [], "axes_lengths": [], "orientations": [], "angles_deviation": []}

    # Calcul de référence (sans perturbation)
    center_of_mass = compute_center_of_mass(points)
    normalized_points, P = normalize_points(points, center_of_mass)
    K = compute_k_matrix(normalized_points)
    q = run_douglas_rachford(K)
    A, b, c = extract_parameters(q)
    A, b, c = denormalize_parameters(A, b, c, center_of_mass, P)
    centroid = calculate_centroid(A, b) - center_of_mass
    eigvals, eigvecs = np.linalg.eigh(A)
    axes_lengths = np.sqrt(1 / eigvals)

    for i in range(n_simulations):
        # Ajouter du bruit gaussien aux points
        #perturbed_points = points + np.random.normal(0, noise_std, points.shape)

        # Définition des tailles des voxels (exemple : 0.2 µm en XY, 1 µm en Z)
        voxel_size = (0.5, 0.189, 0.189)  # Mettre tes valeurs réelles ici

        # Génération des points perturbés avec correction d’anisotropie
        perturbed_points = perturb_points_anisotropic(points, noise_std, voxel_size, isotropic)

        # Pipeline d'ajustement
        center_of_mass = compute_center_of_mass(perturbed_points)
        normalized_points, P = normalize_points(perturbed_points, center_of_mass)
        K = compute_k_matrix(normalized_points)
        q = run_douglas_rachford(K)
        A, b, c = extract_parameters(q)
        A, b, c = denormalize_parameters(A, b, c, center_of_mass, P)
        centroid_perturbed = calculate_centroid(A, b) - center_of_mass
        eigvals_perturbed, eigvecs_perturbed = np.linalg.eigh(A)
        axes_lengths_perturbed = np.sqrt(1 / eigvals_perturbed)

        # Calculer la déviation angulaire des axes X, Y, Z
        angles_deviation = compute_orientation_deviation(eigvecs, eigvecs_perturbed)

        # Enregistrer les résultats
        results["centroids"].append(centroid_perturbed)
        results["axes_lengths"].append(axes_lengths_perturbed)
        results["orientations"].append(eigvecs_perturbed)
        results["angles_deviation"].append(angles_deviation)

    return results

def test_sensitivity_multiple_std(points, PA, PD, std_values, n_simulations=50, voxel_size = (0.5, 0.455, 0.455),
                                  isotropic=True):
    """
    Teste plusieurs niveaux de bruit et stocke les résultats pour analyse.

    Args:
        points (np.ndarray): Points d'origine.
        std_values (list): Liste des écarts-types du bruit.
        voxel_size (tuple): Taille des voxels en (dx, dy, dz).
        n_simulations (int): Nombre de simulations par niveau de bruit.

    Returns:
        dict: Résultats pour chaque niveau de bruit.
    """
    results = {std: {"centroids": [], "axes_lengths": [], "orientations": [],
                     "angles_deviation": []} for std in std_values}

    # Calcul de référence (sans perturbation)
    center_of_mass = compute_center_of_mass(points)
    normalized_points, P = normalize_points(points, center_of_mass)
    K = compute_k_matrix(normalized_points)
    q = run_douglas_rachford(K)
    A, b, c = extract_parameters(q)
    A, b, c = denormalize_parameters(A, b, c, center_of_mass, P)
    centroid = calculate_centroid(A, b) - center_of_mass
    eigvals, eigvecs = np.linalg.eigh(A)

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2] = orienter_axes(eigvecs,PA, PD, centroid)

    axes_lengths = np.sqrt(1 / eigvals)

    for std in std_values:
        for i in range(n_simulations):
            # Ajouter du bruit gaussien aux points
            # perturbed_points = points + np.random.normal(0, noise_std, points.shape)

            # Génération des points perturbés avec correction d’anisotropie
            perturbed_points = perturb_points_anisotropic(points, std, voxel_size, isotropic)

            # Pipeline d'ajustement
            center_of_mass = compute_center_of_mass(perturbed_points)
            normalized_points, P = normalize_points(perturbed_points, center_of_mass)
            K = compute_k_matrix(normalized_points)
            q = run_douglas_rachford(K)
            A, b, c = extract_parameters(q)
            A, b, c = denormalize_parameters(A, b, c, center_of_mass, P)
            centroid_perturbed = calculate_centroid(A, b) - center_of_mass
            eigvals_perturbed, eigvecs_perturbed = np.linalg.eigh(A)

            order = np.argsort(eigvals_perturbed)  # Bien que eigh renvoie déjà dans l'ordre croissant
            eigvals_perturbed = eigvals_perturbed[order]
            eigvecs_perturbed = eigvecs_perturbed[:, order]

            eigvecs_perturbed[:, 0], eigvecs_perturbed[:, 1], eigvecs_perturbed[:, 2] = orienter_axes(eigvecs_perturbed,
                                                                                        PA, PD, centroid_perturbed)

            axes_lengths_perturbed = np.sqrt(1 / eigvals_perturbed)

            # Calculer la déviation angulaire des axes X, Y, Z
            angles_deviation = compute_orientation_deviation(eigvecs, eigvecs_perturbed)

            # Stocker les erreurs
            # Enregistrer les résultats
            results[std]["centroids"].append(centroid_perturbed)
            results[std]["axes_lengths"].append(axes_lengths_perturbed)
            results[std]["orientations"].append(eigvecs_perturbed)
            results[std]["angles_deviation"].append(angles_deviation)

    return results, eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]


def add_perturbed_centroids_to_napari(viewer, perturbed_centroids):
    """
    Ajoute un layer de points dans Napari pour visualiser les centroïdes perturbés.

    Args:
        viewer (napari.Viewer): Viewer Napari en cours.
        perturbed_centroids (np.ndarray): Liste des centroïdes perturbés (N,3).
    """
    viewer.add_points(
        perturbed_centroids,
        name="Centroïdes perturbés",
        size=5,
        face_color="blue",
        opacity=0.6
    )
    print(f"✅ {len(perturbed_centroids)} centroïdes perturbés ajoutés dans Napari")


def transformer_coordonnees(point, origine, u, v, w):
    transformation_matrix = np.array([u, v, w]).T
    translation = point - origine
    new_coords = np.linalg.solve(transformation_matrix, translation[0, :])
    return new_coords


def compute_point_distances_in_perturbed_references(reference_point, original_centroid, u, v, w, results, voxel_size):
    """
    Calcule la distance du point de référence dans chaque repère perturbé.

    Args:
        reference_point (np.ndarray): Coordonnées du point fixé (1,3).
        original_centroid (np.ndarray): Centroïde du repère original.
        original_axes (np.ndarray): Longueurs principales du repère original.
        results (dict): Résultats des simulations pour chaque std.
        voxel_size (tuple): Taille des voxels (dz, dy, dx).

    Returns:
        list: Liste des distances du point fixé dans chaque repère perturbé.
    """
    distances_per_std = []
    reference_coords = transformer_coordonnees(reference_point, original_centroid, u, v, w)
    perturbed_coords_per_std = []
    for std in results.keys():
        distances = []
        perturbed_coords_list = []
        for i in range(len(results[std]["centroids"])):
            perturbed_centroid = np.array(results[std]["centroids"][i])
            perturbed_u = np.array(results[std]["orientations"][i])[:, 0]
            perturbed_v = np.array(results[std]["orientations"][i])[:, 1]
            perturbed_w = np.array(results[std]["orientations"][i])[:, 2]

            perturbed_coords = transformer_coordonnees(reference_point, perturbed_centroid, perturbed_u, perturbed_v,
                                                       perturbed_w)
            perturbed_coords_list.append(perturbed_coords)
            # Calcul de la distance entre les points dans les repères perturbés
            distance = np.linalg.norm((perturbed_coords - reference_coords) * voxel_size)
            distances.append(distance)

        distances_per_std.append(distances)
        perturbed_coords_per_std.append(perturbed_coords_list)

    return distances_per_std, reference_coords, perturbed_coords_per_std

def compute_point_distances_in_perturbed_references_pts_nbre_xp(reference_point, original_centroid, u, v, w, results, voxel_size):
    """
    Calcule la distance du point de référence dans chaque repère perturbé.

    Args:
        reference_point (np.ndarray): Coordonnées du point fixé (1,3).
        original_centroid (np.ndarray): Centroïde du repère original.
        results (dict): Résultats des simulations pour chaque std.
        voxel_size (tuple): Taille des voxels (dz, dy, dx).

    Returns:
        list: Liste des distances du point fixé dans chaque repère perturbé.
    """
    distances_per_std = []
    reference_coords = transformer_coordonnees(reference_point, original_centroid, u, v, w)
    perturbed_coords_per_std = []

    for std in results.keys():
        distances = []
        perturbed_coords_list = []

        # On boucle sur les m_simulations
        for i in range(len(results[std]["centroids"])):
            for j in range(len(results[std]["centroids"][i])):  # On traite chaque centroid individuellement
                perturbed_centroid = np.array(results[std]["centroids"][i][j])  # Shape (3,)
                perturbed_u = np.array(results[std]["orientations"][i][j])[:, 0]
                perturbed_v = np.array(results[std]["orientations"][i][j])[:, 1]
                perturbed_w = np.array(results[std]["orientations"][i][j])[:, 2]

                # Correction: traiter chaque centroid individuellement
                perturbed_coords = transformer_coordonnees(reference_point, perturbed_centroid, perturbed_u, perturbed_v, perturbed_w)
                perturbed_coords_list.append(perturbed_coords)

                # Calcul de la distance entre les points dans les repères perturbés
                distance = np.linalg.norm((perturbed_coords - reference_coords) * voxel_size)
                distances.append(distance)

        distances_per_std.append(distances)
        perturbed_coords_per_std.append(perturbed_coords_list)

    return distances_per_std, reference_coords, perturbed_coords_per_std


def analyze_results(original_centroid, original_axes, results):
    """
    Analyse des résultats de sensibilité.

    Args:
        original_centroid (np.ndarray): Centroïde d'origine.
        original_axes (np.ndarray): Longueurs des axes d'origine.
        results (dict): Résultats des simulations.
    """
    perturbed_centroids = np.array(results["centroids"])
    perturbed_axes = np.array(results["axes_lengths"])
    perturbed_angles = np.array(results["orientations"])

    # 📏 Calcul des déviations
    centroid_deviation = np.linalg.norm(perturbed_centroids - original_centroid, axis=1)
    axes_deviation = np.abs(perturbed_axes - original_axes)

    # 📊 Mesures quantitatives
    mean_centroid_deviation = np.mean(centroid_deviation)
    std_centroid_deviation = np.std(centroid_deviation)
    max_centroid_deviation = np.max(centroid_deviation)
    min_centroid_deviation = np.min(centroid_deviation)

    print(f"📊 Distance moyenne du centroïde : {mean_centroid_deviation:.4f} ± {std_centroid_deviation:.4f}")
    print(f"📏 Min: {min_centroid_deviation:.4f}, Max: {max_centroid_deviation:.4f}")

    # 🔵 Graphique 1 : Déplacement des centroïdes avec vecteurs
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")

    # Centroïde original
    ax1.scatter(*original_centroid, color="red", label="Original", s=100)

    # Centroïdes perturbés + Vecteurs de déplacement
    for i in range(len(perturbed_centroids)):
        ax1.quiver(*original_centroid,
                   *(perturbed_centroids[i] - original_centroid),
                   color="black", alpha=0.5, arrow_length_ratio=0.2)

    distances = np.linalg.norm(perturbed_centroids - original_centroid, axis=1)
    sc = ax1.scatter(perturbed_centroids[:, 0], perturbed_centroids[:, 1], perturbed_centroids[:, 2],
                     c=distances, cmap='coolwarm', alpha=0.7, s=10, label="Perturbés")

    fig.colorbar(sc, ax=ax1, label="Distance au centroïde original")
    ax1.set_title("Déplacement des centroïdes (flèches)")
    ax1.legend()

    # 🟠 Graphique 2 : Déviation des longueurs des axes
    ax2 = fig.add_subplot(122)
    for i, axe in zip(range(3), ["Axe Z", "Axe Y", "Axe X"]):
        ax2.plot(range(len(axes_deviation)), axes_deviation[:, i], label=axe)
    ax2.set_title("Déviation des longueurs d'axes")
    ax2.set_xlabel("Simulation")
    ax2.set_ylabel("Déviation")
    ax2.legend()
    plt.show()

    # 📈 Graphique 3 : Distance du centroïde
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(centroid_deviation)), centroid_deviation, marker='o', linestyle='-', color='black')
    plt.axhline(mean_centroid_deviation, color='red', linestyle="--", label="Moyenne")
    plt.xlabel("Simulation")
    plt.ylabel("Distance au centroïde original")
    plt.title("Évolution du déplacement du centroïde")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 🟢 Graphique 3 : Déviation angulaire des axes
    plot_orientation_deviation(results["angles_deviation"])

    # 🟣 Graphique 4 : Histogramme des déplacements
    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=20, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel("Distance au centroïde original")
    plt.ylabel("Nombre d'occurrences")
    plt.title("Distribution des déplacements du centroïde")
    plt.grid(True)
    plt.show()


def inverse_transformer_coordonnees(x_pert, origine, u, v, w):
    """
    Transforme les coordonnées x_pert (dans le repère perturbé)
    vers le repère original, sachant que le repère perturbé
    est défini par son origine 'origine' et les axes u, v, w.

    Paramètres :
      - x_pert : coordonnées dans le repère perturbé (vecteur 1D de taille 3)
      - origine: centroïde du repère perturbé (vecteur 1D de taille 3)
      - u, v, w : axes (vecteurs 1D de taille 3) du repère perturbé

    Retourne :
      - x_orig : coordonnées du point dans le repère original.
    """
    T = np.column_stack((u, v, w))  # Matrice de passage du repère perturbé vers le repère original
    x_orig = origine + T @ x_pert
    return x_orig


def analyze_results_multiple_std(original_centroid, u, v, w, original_axes, results, voxel_size, reference_point,
                                 viewer):
    """
    Analyse et affiche les résultats de sensibilité pour plusieurs valeurs de bruit sous forme de boxplots.

    Args:
        original_centroid (np.ndarray): Centroïde d'origine.
        original_axes (np.ndarray): Longueurs des axes d'origine.
        results (dict): Résultats des simulations pour chaque valeur de bruit.
    """
    std_values = list(results.keys())

    # 📌 Déviation du centroïde en µm
    centroid_errors_um = [
        np.linalg.norm((np.array(results[std]["centroids"]) - original_centroid) * voxel_size, axis=1)
        for std in std_values
    ]
    µm_std_values = [x * voxel_size[1] for x in std_values]
    plt.figure(figsize=(8, 5))
    plt.boxplot(centroid_errors_um, labels=[f"{x:.1f}" for x in µm_std_values])
    plt.xlabel("Positioning error (µm)")
    plt.ylabel("Displacement of the origin of the reference frame (µm)")
    plt.title("Impact of positioning error on origin position (µm)")
    plt.grid(True)
    plt.show()

    # 📌 Déviation des longueurs d'axes (X, Y, Z)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Noms des axes
    axis_labels = ["Axe Z", "Axe Y", "Axe X"]

    # Liste de couleurs pour chaque std
    colors = plt.cm.plasma(np.linspace(0, 1, len(std_values)))

    # Définition des positions X pour chaque axe (Z=1, Y=2, X=3)
    x_positions = np.array([1, 2, 3])

    # Convertir les std en µm pour l'affichage
    µm_std_values = [x * voxel_size[1] for x in std_values]

    for i, std in enumerate(std_values):
        # Calcul des déviations des longueurs des axes
        data_um = [
            (np.array(results[std]["axes_lengths"])[:, j] - original_axes[j]) * voxel_size[j]
            for j in range(3)
        ]

        # Ajout du boxplot avec un léger décalage horizontal
        ax.boxplot(data_um, positions=x_positions + (i - len(std_values) / 2) * 0.1, widths=0.08,
                   patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.7))

    # Mise en forme du graphique
    ax.set_xticks(x_positions)
    ax.set_xticklabels(axis_labels)
    ax.set_xlabel("Ellipsoid axes")
    ax.set_ylabel("Axes length variation (µm)")
    ax.set_title("Impact of the positioning error on the axes length (in µm)")

    # Légende des couleurs (STD values en µm)
    handles = [plt.Line2D([0], [0], color=colors[i], lw=4, label=f"{µm_std_values[i]:.1f} µm")
               for i in range(len(std_values))]
    ax.legend(handles=handles, title="Positioning error (µm)")

    ax.grid(True)
    plt.show()

    # 📌 Déviation angulaire des axes (X, Y, Z)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Noms des axes
    axis_labels = ["Axe Z", "Axe Y", "Axe X"]

    # Liste de couleurs pour chaque std
    colors = plt.cm.plasma(np.linspace(0, 1, len(std_values)))

    # Définition des positions X pour chaque axe (Z=1, Y=2, X=3)
    x_positions = np.array([1, 2, 3])

    for i, std in enumerate(std_values):
        # Assurer que chaque élément est bien de taille (N, 3)
        data = [np.array(results[std]["angles_deviation"]).squeeze()[:, j] for j in range(3)]

        # Ajout du boxplot avec un léger décalage horizontal
        ax.boxplot(data, positions=x_positions + (i - len(std_values) / 2) * 0.1, widths=0.08,
                   patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.7))

    # Mise en forme du graphique
    ax.set_xticks(x_positions)
    ax.set_xticklabels(axis_labels)
    ax.set_xlabel("Main axes of the ellipsoid")
    ax.set_ylabel("Angular deviation (°)")
    ax.set_title("Impact of the positioning error on axes orientation")

    # Légende des couleurs (STD values)
    handles = [plt.Line2D([0], [0], color=colors[i], lw=4, label=f"{µm_std_values[i]:.1f} µm")
               for i in range(len(std_values))]
    ax.legend(handles=handles, title="Positioning error (µm)")

    ax.grid(True)
    plt.show()

    # 🔹 Calcul des distances du point fixé dans les repères perturbés
    distances, reference_coords, perturbed_coords_per_std = compute_point_distances_in_perturbed_references(
                                                            reference_point, original_centroid, u, v, w, results,
                                                            voxel_size)

    # 📌 Boxplot des distances du point dans les repères perturbés
    plt.figure(figsize=(8, 5))
    plt.boxplot(distances, tick_labels=[f"{x:.1f}" for x in µm_std_values])
    plt.xlabel("Positioning error (µm)")
    plt.ylabel("Distance from set point (µm)")
    plt.title("Impact of positioning error on the reference point location in disturbed reference frames")
    plt.grid(True)
    plt.show()

    # 📌 Visualisation 3D des perturbations pour chaque std
    # Pour chaque valeur de bruit (std) :
    # 1. Calcul du rayon maximal global (en microns)
    global_max_radius = 0
    radius_list = []
    for i, std in enumerate(std_values):
        pts = np.array(perturbed_coords_per_std[i])
        # Conversion en microns
        pts_microns = pts * np.array(voxel_size)
        ref_microns = reference_coords * np.array(voxel_size)
        # Coordonnées relatives : le point de référence sera (0,0,0)
        pts_rel = pts_microns - ref_microns
        max_radius = np.max(np.linalg.norm(pts_rel, axis=1))
        radius_list.append(max_radius)
        if max_radius > global_max_radius:
            global_max_radius = max_radius

    # 2. Création d'une figure regroupant toutes les sous-figures (ici, 1 ligne x 5 colonnes)
    fig = plt.figure(figsize=(20, 5))
    num_plots = len(std_values)

    for i, std in enumerate(std_values):
        ax = fig.add_subplot(1, num_plots, i + 1, projection="3d")

        # Conversion en microns
        ref_microns = reference_coords * np.array(voxel_size)
        pts = np.array(perturbed_coords_per_std[i])
        pts_microns = pts * np.array(voxel_size)
        pts_rel = pts_microns - ref_microns  # On recentre : le point de référence devient (0,0,0)

        # Affichage du point de référence en rouge à l'origine
        ax.scatter(0, 0, 0, color="red", label="Reference (0,0,0)", s=100)

        # Affichage des points perturbés (coordonnées relatives) en bleu
        ax.scatter(pts_rel[:, 0], pts_rel[:, 1], pts_rel[:, 2],
                   color="blue", alpha=0.6, label="Perturbed points")

        # Calculer le rayon maximum pour ce sous-ensemble (optionnel, ici on utilise global_max_radius)
        # max_radius = np.max(np.linalg.norm(pts_rel, axis=1))
        # On utilise le global_max_radius pour imposer la même échelle partout.

        # Création d'une sphère centrée sur (0,0,0) avec le rayon global
        U, V = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        X = radius_list[i] * np.cos(U) * np.sin(V)
        Y = radius_list[i] * np.sin(U) * np.sin(V)
        Z = radius_list[i] * np.cos(V)
        ax.plot_wireframe(X, Y, Z, color="orange", alpha=0.3)

        # Fixer les limites des axes pour avoir le même échelle sur X, Y, Z
        ax.set_xlim([-global_max_radius, global_max_radius])
        ax.set_ylim([-global_max_radius, global_max_radius])
        ax.set_zlim([-global_max_radius, global_max_radius])

        ax.set_title(f"Positioning error: {µm_std_values[i]:.1f} µm")
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        ax.legend()

    #plt.tight_layout()
    plt.show()

    # Calculez le nuage des positions du point de référence dans le repère original
    points_ref_orig = []  # liste qui contiendra les positions ramenées dans le repère original

    # On parcourt tous les niveaux de bruit

    for std in sorted(results.keys()):
        points_ref_orig = []  # Réinitialiser la liste pour ce niveau de bruit
        for i in range(len(results[std]["centroids"])):
            # Récupération du repère perturbé pour cette simulation
            perturbed_centroid = np.array(results[std]["centroids"][i])
            perturbed_axes = np.array(results[std]["orientations"][i])
            # On suppose que les colonnes correspondent aux axes dans l'ordre souhaité
            perturbed_u = perturbed_axes[:, 0]
            perturbed_v = perturbed_axes[:, 1]
            perturbed_w = perturbed_axes[:, 2]

            # Calculer les coordonnées du point de référence dans le repère perturbé
            # Ici, transformer_coordonnees utilise le repère de base (original_centroid, u, v, w)
            # qui devrait idéalement renvoyer [0,0,0] si le point était exactement positionné.
            x_pert = transformer_coordonnees(reference_point, original_centroid, u, v, w)

            # "Projeter" cet écart dans le repère d'origine du pertubé à l'aide de inverse_transformer_coordonnees.
            x_orig = inverse_transformer_coordonnees(x_pert, perturbed_centroid, perturbed_u, perturbed_v, perturbed_w)
            points_ref_orig.append(x_orig)

        points_ref_orig = np.array(points_ref_orig)
        # Créer un layer Napari pour ce niveau de bruit, en affichant les points en jaune
        viewer.add_points(points_ref_orig,
                          name=f"Points ref pert STD {std:.2f}",
                          size=5,
                          face_color="yellow")

        # Choisir, par exemple, la première simulation pour dessiner les axes représentatifs
        rep_centroid = np.array(results[std]["centroids"][0])
        rep_axes = np.array(results[std]["orientations"][0])
        rep_u = rep_axes[:, 0]
        rep_v = rep_axes[:, 1]
        rep_w = rep_axes[:, 2]

        # Appeler draw_axes avec un label indiquant le niveau de bruit
        draw_axes(viewer, rep_centroid, rep_u, rep_v, rep_w, std_label=f"STD {std:.2f}")

def test_sensitivity_subset(points, PA, PD, subset_ratios, std_values, n_simulations=50, m_noise_simulations=50,
                            voxel_size=(0.5, 0.455, 0.455), isotropic=True):
    """
    Teste l'impact combiné du nombre de points et du bruit de positionnement.

    Args:
        points (np.ndarray): Points d'origine.
        PA, PD (np.ndarray): Points de référence pour l'orientation.
        subset_ratios (list): Liste des pourcentages de points utilisés.
        std_values (list): Liste des niveaux de bruit testés.
        n_simulations (int): Nombre de répétitions pour le choix des points.
        m_noise_simulations (int): Nombre de répétitions pour l'ajout de bruit.
        voxel_size (tuple): Taille des voxels.
        isotropic (bool): Si True, bruit isotrope.

    Returns:
        dict: Résultats moyens pour chaque (ratio, std).
    """
    subset_results = {
        ratio: {
            std: {
                "centroids": [[] for _ in range(n_simulations)],  # Liste de listes pour séparer les N simulations
                "axes_lengths": [[] for _ in range(n_simulations)],
                "angles_deviation": [[] for _ in range(n_simulations)],
                "orientations": [[] for _ in range(n_simulations)]
            }
            for std in std_values
        }
        for ratio in subset_ratios
    }

    full_center_of_mass = compute_center_of_mass(points)

    for ratio in subset_ratios:
        for std in std_values:
            for i in range(n_simulations):  # Itération sur les choix de points
                subset_points = points[np.random.choice(points.shape[0], int(ratio * points.shape[0]), replace=False)]

                # Calcul de référence
                full_center_of_mass = compute_center_of_mass(subset_points)
                normalized_points, P = normalize_points(subset_points, full_center_of_mass)
                K = compute_k_matrix(normalized_points)
                q = run_douglas_rachford(K)
                A, b, c = extract_parameters(q)
                A, b, c = denormalize_parameters(A, b, c, full_center_of_mass, P)
                centroid_ref = calculate_centroid(A, b) - full_center_of_mass
                eigvals, eigvecs = np.linalg.eigh(A)

                order = np.argsort(eigvals)
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]

                eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2] = orienter_axes(eigvecs, PA, PD, centroid_ref)

                for _ in range(m_noise_simulations):  # Itération sur les perturbations de bruit
                    perturbed_points = perturb_points_anisotropic(subset_points, std, voxel_size, isotropic)

                    # Pipeline de fit
                    center_of_mass = compute_center_of_mass(perturbed_points)
                    normalized_points, P = normalize_points(perturbed_points, center_of_mass)
                    K = compute_k_matrix(normalized_points)
                    q = run_douglas_rachford(K)
                    A, b, c = extract_parameters(q)
                    A, b, c = denormalize_parameters(A, b, c, center_of_mass, P)
                    centroid_perturbed = calculate_centroid(A, b) - full_center_of_mass
                    eigvals_perturbed, eigvecs_perturbed = np.linalg.eigh(A)

                    order = np.argsort(eigvals_perturbed)
                    eigvals_perturbed = eigvals_perturbed[order]
                    eigvecs_perturbed = eigvecs_perturbed[:, order]

                    eigvecs_perturbed[:, 0], eigvecs_perturbed[:, 1], eigvecs_perturbed[:, 2] = orienter_axes(
                        eigvecs_perturbed, PA, PD, centroid_perturbed)

                    axes_lengths_perturbed = np.sqrt(1 / eigvals_perturbed)
                    angles_deviation = compute_orientation_deviation(eigvecs, eigvecs_perturbed)

                    # On ajoute à la liste correspondant à la simulation i
                    subset_results[ratio][std]["centroids"][i].append(centroid_perturbed)
                    subset_results[ratio][std]["axes_lengths"][i].append(axes_lengths_perturbed)
                    subset_results[ratio][std]["angles_deviation"][i].append(angles_deviation)
                    subset_results[ratio][std]["orientations"][i].append(eigvecs_perturbed)

    return subset_results

def parallel_test_sensitivity_subset(points, PA, PD, subset_ratios, std_values, n_simulations=50, m_noise_simulations=50,
                                     voxel_size=(0.5, 0.455, 0.455), isotropic=True, n_jobs=-1):
    """
    Version parallèle de test_sensitivity_subset.
    """
    subset_results = {
        ratio: {
            std: {
                "centroids": [[] for _ in range(n_simulations)],
                "axes_lengths": [[] for _ in range(n_simulations)],
                "angles_deviation": [[] for _ in range(n_simulations)],
                "orientations": [[] for _ in range(n_simulations)]
            }
            for std in std_values
        }
        for ratio in subset_ratios
    }

    full_center_of_mass = compute_center_of_mass(points)

    def process_one_simulation(ratio, std, i):
        """
        Effectue une simulation (choix d'un sous-ensemble et bruitage).
        """
        subset_points = points[np.random.choice(points.shape[0], int(ratio * points.shape[0]), replace=False)]

        # Calcul de référence
        center_of_mass = compute_center_of_mass(subset_points)
        normalized_points, P = normalize_points(subset_points, center_of_mass)
        K = compute_k_matrix(normalized_points)
        q = run_douglas_rachford(K)
        A, b, c = extract_parameters(q)
        A, b, c = denormalize_parameters(A, b, c, center_of_mass, P)
        centroid_ref = calculate_centroid(A, b) - full_center_of_mass
        eigvals, eigvecs = np.linalg.eigh(A)

        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2] = orienter_axes(eigvecs, PA, PD, centroid_ref)

        results = {
            "centroids": [],
            "axes_lengths": [],
            "angles_deviation": [],
            "orientations": []
        }

        for _ in range(m_noise_simulations):
            perturbed_points = perturb_points_anisotropic(subset_points, std, voxel_size, isotropic)

            # Pipeline de fit
            center_of_mass = compute_center_of_mass(perturbed_points)
            normalized_points, P = normalize_points(perturbed_points, center_of_mass)
            K = compute_k_matrix(normalized_points)
            q = run_douglas_rachford(K)
            A, b, c = extract_parameters(q)
            A, b, c = denormalize_parameters(A, b, c, center_of_mass, P)
            centroid_perturbed = calculate_centroid(A, b) - full_center_of_mass
            eigvals_perturbed, eigvecs_perturbed = np.linalg.eigh(A)

            order = np.argsort(eigvals_perturbed)
            eigvals_perturbed = eigvals_perturbed[order]
            eigvecs_perturbed = eigvecs_perturbed[:, order]

            eigvecs_perturbed[:, 0], eigvecs_perturbed[:, 1], eigvecs_perturbed[:, 2] = orienter_axes(
                eigvecs_perturbed, PA, PD, centroid_perturbed)

            axes_lengths_perturbed = np.sqrt(1 / eigvals_perturbed)
            angles_deviation = compute_orientation_deviation(eigvecs, eigvecs_perturbed)

            # Stocker les résultats localement
            results["centroids"].append(centroid_perturbed)
            results["axes_lengths"].append(axes_lengths_perturbed)
            results["angles_deviation"].append(angles_deviation)
            results["orientations"].append(eigvecs_perturbed)

        return ratio, std, i, results

    # Lancer en parallèle
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(process_one_simulation)(ratio, std, i)
        for ratio in subset_ratios
        for std in std_values
        for i in range(n_simulations)
    )

    # Stocker proprement les résultats
    for ratio, std, i, results in parallel_results:
        subset_results[ratio][std]["centroids"][i] = results["centroids"]
        subset_results[ratio][std]["axes_lengths"][i] = results["axes_lengths"]
        subset_results[ratio][std]["angles_deviation"][i] = results["angles_deviation"]
        subset_results[ratio][std]["orientations"][i] = results["orientations"]

    return subset_results


def analyze_results_subset(subset_results, original_centroid, original_axes, voxel_size, u, v, w, reference_point,
                           m_noise_simulations):
    """
    Analyse l'impact combiné du nombre de points et du bruit sur la stabilité des résultats.

    Args:
        subset_results (dict): Résultats de test_sensitivity_subset.
        original_centroid (np.ndarray): Centroïde d'origine.
        original_axes (np.ndarray): Longueurs des axes d'origine.
        voxel_size (tuple): Taille des voxels.
    """
    subset_ratios = sorted(subset_results.keys())
    std_values = sorted(subset_results[subset_ratios[0]].keys())

    colors = plt.cm.plasma(np.linspace(0, 1, len(std_values)))

    # 📌 Déviation du centroïde en µm (moyenne des moyennes)
    plt.figure(figsize=(8, 5))
    for i, std in enumerate(std_values):
        centroid_errors_um = [
            np.mean([  # Moyenne sur N tirages de points
                np.mean([  # Moyenne sur M bruitages
                    np.linalg.norm((np.array(centroid) - original_centroid) * voxel_size)
                    for centroid in subset_results[r][std]["centroids"][i]  # Liste des perturbations M
                ]) if subset_results[r][std]["centroids"][i] else 0  # Gérer les cas vides
                for i in range(len(subset_results[r][std]["centroids"]))  # Moyenne sur N choix de points
            ])
            for r in subset_ratios
        ]

        plt.plot([int(r * 89) for r in subset_ratios], centroid_errors_um, marker='o', linestyle='-',
                 label=f"{std * voxel_size[1]:.1f} µm", color=colors[i])

    plt.xlabel("Total points used")
    plt.ylabel("Mean displacement of the origin (µm)")
    plt.title("Impact of point count and noise on origin displacement")
    plt.legend(title="Positioning error (µm)")
    plt.grid(True)
    plt.show()

    # 📌 Déviation angulaire moyenne des axes (Z, Y, X) en fonction du nombre de points
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)  # Trois graphes, même échelle en Y

    axis_labels = ["Axe Z", "Axe Y", "Axe X"]

    # Trouver la plage commune pour l'axe Y
    all_deviations = []

    for i, std in enumerate(std_values):
        for axis in range(3):
            deviations = [
                np.mean([  # Moyenne sur N tirages de points
                    np.mean([angles[j][axis] for j in range(len(angles))])  # Moyenne sur M bruitages
                    for angles in subset_results[r][std]["angles_deviation"]
                ])
                for r in subset_ratios
            ]
            all_deviations.extend(deviations)

    y_min, y_max = min(all_deviations), max(all_deviations)  # Valeurs min/max pour normalisation

    # 📌 Tracer un graphe par axe
    for axis, ax in enumerate(axes):
        for i, std in enumerate(std_values):
            mean_angular_deviation = [
                np.mean([  # Moyenne sur N tirages de points
                    np.mean([angles[j][axis] for j in range(len(angles))])  # Moyenne sur M bruitages
                    for angles in subset_results[r][std]["angles_deviation"]
                ])
                for r in subset_ratios
            ]

            ax.plot([int(r * 89) for r in subset_ratios], mean_angular_deviation, marker='o', linestyle='-',
                    label=f"{std * voxel_size[1]:.1f} µm", color=colors[i])

        ax.set_xlabel("Total points used")
        ax.set_ylabel("Mean angular deviation (°)")
        ax.set_title(f"Impact of point count and noise on {axis_labels[axis]}")
        ax.legend(title="Positioning error (µm)")
        ax.grid(True)
        ax.set_ylim(y_min, y_max)  # Assurer une échelle commune en Y

    plt.tight_layout()
    plt.show()

    for i, std in enumerate(std_values):
        mean_distances = []

        for r in subset_ratios:
            distances_for_ratio = []

            # 🔹 Convertir les résultats pour le format attendu
            std_results = {
                std: {
                    "centroids": subset_results[r][std]["centroids"],
                    "axes_lengths": subset_results[r][std]["axes_lengths"],
                    "orientations": subset_results[r][std]["orientations"],
                    "angles_deviation": subset_results[r][std]["angles_deviation"],
                }
            }

            for j in range(len(subset_results[r][std]["centroids"])):  # Parcourt les N tirages de points
                distances, _, _ = compute_point_distances_in_perturbed_references_pts_nbre_xp(
                    reference_point, original_centroid, u, v, w, std_results, voxel_size
                )

                # 🔹 Prendre la moyenne sur M tirages de bruit
                mean_distance_per_noise = [np.mean(dist) for dist in distances[0]]

                # 🔹 Ensuite, stocker la moyenne des M pour cette simulation (N)
                distances_for_ratio.append(np.mean(mean_distance_per_noise))

            # 🔹 Enfin, moyenne sur les N simulations
            mean_distances.append(np.mean(distances_for_ratio))

        plt.plot([int(r * 89) for r in subset_ratios], mean_distances, marker='o', linestyle='-',
                 label=f"{std * voxel_size[1]:.1f} µm", color=colors[i])

    plt.xlabel("Total points used")
    plt.ylabel("Mean distance from reference point (µm)")
    plt.title("Impact of point count and noise on the reference point location")
    plt.legend(title="Positioning error (µm)")
    plt.grid(True)
    plt.show()


### END OF SENSITIVITY ANALYSIS ###

def main():
    # Ajout de l'option argparse
    parser = argparse.ArgumentParser(description="Visualisation d'une ellipsoïde ajustée sur des points.")
    parser.add_argument("--mode", type=str, choices=["synthetic", "real"], default="real",
                        help="Choisir entre les points synthétiques ('synthetic') ou réels ('real').")
    parser.add_argument("--multiple_std", "-mstd", type=bool, default=True,
                        help="Calculer les erreurs avec plusieurs niveaux de bruit ou non")
    args = parser.parse_args()
    mode = args.mode
    mstd = args.multiple_std

    if mode == "synthetic":

        # Points ellipse synthétique
        points = np.array([
            [49, 30.26, 49.12],
            [49, 31.87, 55.89],
            [49, 36.61, 60.47],
            [49, 40.77, 63.52],
            [49, 47.46, 64.37],
            [49, 57.29, 64.71],
            [49, 65.77, 58.78],
            [69, 34.66, 48.52],
            [69, 43.73, 39.37],
            [69, 52.89, 38.18],
            [69, 65.01, 46.83],
            [69, 60.85, 58.61],
            [77, 42.04, 47.08],
            [77, 41.44, 49.28],
            [77, 48.14, 56.57],
            [77, 58.39, 50.47],
            [21, 45.00, 48.35],
            [21, 44.49, 49.88],
            [21, 50.94, 53.61],
            [21, 55.43, 50.30],
            [21, 50.17, 46.32],
            [20, 50.00, 49.96]
        ])

    else:
        # Points vrai embryon
        points = np.array([[105.        , 266.08354858, 348.71671905],
           [ 52.        , 264.04824892, 323.47900338],
           [ 52.        , 181.82214304, 218.86460134],
           [105.        , 181.00802318, 185.48568707],
           [116.        , 378.43208929, 238.81053791],
           [105.        , 222.93519598, 384.5379929 ],
           [105.        , 297.83422313, 325.92136296],
           [144.        , 274.22474718, 362.14969675],
           [ 95.        , 279.10946634, 334.87668143],
           [144.        , 305.5683618 , 332.84138178],
           [105.        , 213.57281759, 161.46915119],
           [144.        , 231.48345451, 135.41731566],
           [105.        , 234.33287402, 150.88559301],
           [171.        , 244.91643221, 121.57727804],
           [ 52.        , 316.55897991, 247.76585638],
           [ 95.        , 334.46961684, 135.41731566],
           [ 92.        , 262.4200092 , 140.70909475],
           [ 92.        , 362.55675202, 224.97050029],
           [ 95.        , 310.45308096, 301.09070723],
           [116.        , 334.06255691, 123.20551776],
           [ 95.        , 345.86729488, 144.37263412],
           [116.        , 382.90974852, 178.56566825],
           [ 46.        , 159.84090681, 250.61527589],
           [144.        , 380.06032901, 241.65995743],
           [ 92.        , 364.9991116 , 181.8221477 ],
           [144.        , 155.77030751, 176.5303686 ],
           [ 95.        , 333.65549698, 276.66711142],
           [171.        , 228.634035  , 132.16083622],
           [ 95.        , 193.62688101, 185.892747  ],
           [ 46.        , 282.36594578, 303.12600688],
           [ 95.        , 229.04109493, 159.84091147],
           [ 92.        , 365.81323146, 205.83868357],
           [144.        , 186.7068622 , 161.87621112],
           [ 46.        , 234.73993396, 336.09786122],
           [105.        , 200.54689982, 172.05270937],
           [116.        , 283.9941855 , 124.01963762],
           [ 92.        , 356.45085307, 163.50445084],
           [171.        , 209.90927822, 144.77969405],
           [144.        , 290.91420431, 348.71671905],
           [ 52.        , 153.32794793, 251.42939575],
           [ 52.        , 297.02010327, 292.94950862],
           [ 95.        , 257.12823011, 352.78731835],
           [144.        , 354.82261334, 109.77254006],
           [144.        , 300.68364264, 104.48076097],
           [ 95.        , 354.00849348, 156.58443203],
           [ 92.        , 354.00849348, 244.50937694],
           [ 46.        , 196.47630052, 219.6787212 ],
           [ 95.        , 278.29534648, 139.89497489],
           [ 52.        , 225.78461549, 194.0339456 ],
           [105.        , 283.18006564, 338.5402208 ],
           [ 46.        , 291.72832417, 191.99864595],
           [105.        , 338.54021614, 280.73771072],
           [ 95.        , 301.90482243, 136.63849545],
           [171.        , 329.17783775, 310.45308562],
           [171.        , 184.67156255, 161.06209126],
           [ 52.        , 217.64341689, 350.34495877],
           [105.        , 113.43607477, 226.59874001],
           [ 95.        , 369.47677083, 208.68810308],
           [105.        , 258.75646983, 140.30203482],
           [ 92.        , 291.72832417, 136.63849545],
           [ 95.        , 242.47407263, 360.52145703],
           [ 95.        , 358.48615272, 237.58935812],
           [105.        , 326.73547817, 131.34671636],
           [105.        , 142.74438974, 212.75870239],
           [ 46.        , 299.05540292, 285.21536995],
           [171.        , 362.14969209, 279.109471  ],
           [ 95.        , 256.31411025, 146.00087384],
           [ 46.        , 311.26720082, 229.85521945],
           [ 95.        , 291.72832417, 320.22252394],
           [ 95.        , 178.97272353, 200.95396441],
           [116.        , 359.70733251, 260.79177414],
           [105.        , 246.137612  , 145.59381391],
           [ 46.        , 200.54689982, 349.53083891],
           [105.        , 321.03663914, 298.64834764],
           [105.        , 358.89321265, 252.65057554],
           [116.        , 365.40617153, 140.30203482],
           [ 38.        , 171.64564478, 252.65057554],
           [ 38.        , 187.11392213, 235.9611184 ],
           [ 38.        , 207.46691864, 217.64342155],
           [ 38.        , 222.12107612, 207.05986336],
           [ 38.        , 244.91643221, 198.91866476],
           [ 38.        , 266.89766844, 194.84806546],
           [ 38.        , 292.94950396, 204.21044385],
           [ 38.        , 302.31188236, 221.30696092],
           [ 38.        , 302.71894229, 255.09293512],
           [ 38.        , 295.79892348, 279.109471  ],
           [ 38.        , 279.51652627, 297.83422778],
           [ 38.        , 256.72117018, 311.67426541],
           [ 30.        , 224.5634357 , 268.52591281]])

    #points = points[np.random.choice(points.shape[0], int(1.0 * points.shape[0]), replace=False)]

    PA = np.array([144., 354.82261334, 109.77254006])
    PD = np.array([ 30., 224.5634357, 268.52591281])

    center_of_mass = compute_center_of_mass(points)
    normalized_points, P = normalize_points(points, center_of_mass)
    K = compute_k_matrix(normalized_points)

    q = run_douglas_rachford(K)
    A, b, c = extract_parameters(q)

    A, b, c = denormalize_parameters(A, b, c, center_of_mass, P)

    centroid = calculate_centroid(A, b) - center_of_mass
    eigvals, eigvecs = np.linalg.eigh(A)

    viewer = napari.Viewer()
    if mode == "synthetic":
        viewer.add_image(imread("/home/clement/Téléchargements/ovoid_volume.tiff"))
        pt_size = 0.5
    else:
        #viewer.add_image(imread("/home/clement/Images/Experiment 1/20240513_OmpYFP_84mKate_O40_05u_80u_12ss_w1CSU-561 Em 593LP_s7.TIF"))
        viewer.add_image(imread("/run/user/1003/gvfs/smb-share:server=cbi-filer-10.cbi.local,share=batut/Team Batut/"
                                "Vincent_Demont/20250131_cldnGFP_84nlsEOS_postPhotoconversion.lif - E5 - C=0.tif"))
        pt_size = 6
    viewer.add_points(points, name="Points", size=pt_size, face_color="red")
    visualize_ellipsoid(A, centroid, eigvecs, viewer)

    voxel_size = (0.5, 0.455, 0.455)
    # Test de sensibilité
    if mstd is False:
        results = test_sensitivity(points, n_simulations=1000, noise_std=2.0, isotropic=True)
        # Analyse des résultats
        analyze_results(centroid, np.sqrt(1 / eigvals), results)
    else:
        dx = 0.455
        µm_std_values = [0.5, 1.0, 1.5, 2.0]
        std_values = [x / dx for x in µm_std_values]
        results, u, v, w = test_sensitivity_multiple_std(points, PA, PD, n_simulations=10000, std_values=std_values,
                                                         voxel_size=voxel_size, isotropic=True)
        # Analyse des résultats
        reference_point = np.array([[136., 159.1073856, 128.1624962]])  # Point de référence
        analyze_results_multiple_std(centroid, u, v, w, np.sqrt(1 / eigvals),
                                     results, voxel_size, reference_point, viewer)

    # Test de sensibilité au bruit en fonction du nombre de points placés
    import time
    start = time.time()
    subset_ratios = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Pourcentages des points à tester
    noise_simulations = 100
    points_nb_simulations = 100
    subset_results = parallel_test_sensitivity_subset(points, PA, PD, subset_ratios, std_values,
                                                      n_simulations=points_nb_simulations,
                                                      m_noise_simulations=noise_simulations,
                                                      voxel_size=voxel_size)

    analyze_results_subset(subset_results, centroid, np.sqrt(1 / eigvals), voxel_size, u, v, w, reference_point,
                           m_noise_simulations=noise_simulations)
    print("temps de calcul: ", time.time() - start)

    # Ajout du point d'origine (centroïde)
    viewer.add_points([centroid], name="Origine", size=5, face_color="blue")
    draw_axes(viewer, centroid, u, v, w)
    viewer.add_points(reference_point, name="Réference Point", size=10, face_color="green")
    viewer.dims.ndisplay = 3
    napari.run()


if __name__ == "__main__":
    main()
