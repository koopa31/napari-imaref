import numpy as np
import napari
from skimage.io import imread
import argparse
import matplotlib.pyplot as plt


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
    segment_u = np.array([origine, origine + 10 * u])
    viewer.add_shapes(segment_u, shape_type='line', edge_width=1, edge_color='blue', name='Axe X')

    # Dessine l'axe y
    segment_v = np.array([origine, origine + 10 * v])
    viewer.add_shapes(segment_v, shape_type='line', edge_width=1, edge_color='red', name='Axe Y')

    # Dessine l'axe z
    segment_w = np.array([origine, origine + 10 * w])
    viewer.add_shapes(segment_w, shape_type='line', edge_width=1, edge_color='green', name='Axe Z')

    viewer.dims.ndisplay = 3

### SENSITIVITY ANALYSIS ###


def compute_orientation_deviation(original_axes, perturbed_axes):
    """
    Calcule la déviation angulaire des axes principaux entre la configuration originale et perturbée.

    Args:
        original_axes (np.ndarray): Matrice (3x3) des vecteurs propres des axes originaux.
        perturbed_axes_list (list): Liste de matrices (3x3) des vecteurs propres perturbés.

    Returns:
        np.ndarray: Matrice (Nx3) des déviations angulaires (en degrés) des trois axes.
    """
    deviations = []
    angles = []
    for i in range(3):  # X, Y, Z
        dot_product = np.clip(np.dot(original_axes[:, i], perturbed_axes[:, i]), -1.0, 1.0)
        angle = np.arccos(dot_product) * (180 / np.pi)  # Convertir en degrés
        angles.append(angle)
    deviations.append(angles)

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
    if isotropic:
        # Générer un bruit gaussien isotrope
        noise = np.random.normal(0, noise_std, points.shape)

        # Appliquer le bruit de manière identique dans toutes les directions
        perturbed_points = points + noise
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

def test_sensitivity_multiple_std(points, std_values, n_simulations=50, isotropic=True):
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
    axes_lengths = np.sqrt(1 / eigvals)

    for std in std_values:
        for i in range(n_simulations):
            # Ajouter du bruit gaussien aux points
            # perturbed_points = points + np.random.normal(0, noise_std, points.shape)

            # Définition des tailles des voxels (exemple : 0.2 µm en XY, 1 µm en Z)
            voxel_size = (0.5, 0.189, 0.189)  # Mettre tes valeurs réelles ici

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
            axes_lengths_perturbed = np.sqrt(1 / eigvals_perturbed)

            # Calculer la déviation angulaire des axes X, Y, Z
            angles_deviation = compute_orientation_deviation(eigvecs, eigvecs_perturbed)

            # Stocker les erreurs
            # Enregistrer les résultats
            results[std]["centroids"].append(centroid_perturbed)
            results[std]["axes_lengths"].append(axes_lengths_perturbed)
            results[std]["orientations"].append(eigvecs_perturbed)
            results[std]["angles_deviation"].append(angles_deviation)

    return results


def plot_results_boxplots(results):
    """
    Affiche les résultats sous forme de boxplots pour comparer les différentes valeurs de std.

    Args:
        results (dict): Résultats des simulations.
    """
    std_values = list(results.keys())

    # 📌 Déviation du centroïde
    centroid_errors = [results[std]["centroid_errors"] for std in std_values]

    plt.figure(figsize=(8, 5))
    plt.boxplot(centroid_errors, labels=std_values)
    plt.xlabel("Écart-type du bruit")
    plt.ylabel("Déviation du centroïde")
    plt.title("Impact du bruit sur la position du centroïde")
    plt.grid(True)
    plt.show()

    # 📌 Déviation des longueurs d'axes
    fig, ax = plt.subplots(figsize=(8, 5))
    axes_errors_data = {std: [[results[std]["axes_errors"][sim][i] for sim in range(len(results[std]["axes_errors"]))]
                              for i in range(3)]
                        for std in std_values}

    for i, label, color in zip(range(3), ["Axe X", "Axe Y", "Axe Z"], ["red", "blue", "green"]):
        ax.boxplot([axes_errors_data[std][i] for std in std_values], positions=np.array(std_values) + (i - 1) * 0.1,
                   widths=0.1, patch_artist=True, boxprops=dict(facecolor=color, alpha=0.5))

    ax.set_xlabel("Écart-type du bruit")
    ax.set_ylabel("Déviation des longueurs d'axes")
    ax.set_title("Impact du bruit sur la taille des axes")
    ax.legend(["X", "Y", "Z"])
    ax.grid(True)
    plt.show()

    # 📌 Déviation angulaire des axes
    fig, ax = plt.subplots(figsize=(8, 5))
    orientation_errors_data = {std: [[results[std]["orientation_errors"][sim][i] for sim in range(len(results[std]["orientation_errors"]))]
                                     for i in range(3)]
                               for std in std_values}

    for i, label, color in zip(range(3), ["Axe X", "Axe Y", "Axe Z"], ["red", "blue", "green"]):
        ax.boxplot([orientation_errors_data[std][i] for std in std_values], positions=np.array(std_values) + (i - 1) * 0.1,
                   widths=0.1, patch_artist=True, boxprops=dict(facecolor=color, alpha=0.5))

    ax.set_xlabel("Écart-type du bruit")
    ax.set_ylabel("Déviation angulaire (°)")
    ax.set_title("Impact du bruit sur l'orientation des axes")
    ax.legend(["X", "Y", "Z"])
    ax.grid(True)
    plt.show()



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

def analyze_results_multiple_std(original_centroid, original_axes, results):
    """
    Analyse et affiche les résultats de sensibilité pour plusieurs valeurs de bruit sous forme de boxplots.

    Args:
        original_centroid (np.ndarray): Centroïde d'origine.
        original_axes (np.ndarray): Longueurs des axes d'origine.
        results (dict): Résultats des simulations pour chaque valeur de bruit.
    """
    std_values = list(results.keys())

    # 📌 Déviation du centroïde
    centroid_errors = [np.linalg.norm(np.array(results[std]["centroids"]) - original_centroid, axis=1) for std in std_values]

    plt.figure(figsize=(8, 5))
    plt.boxplot(centroid_errors, labels=std_values)
    plt.xlabel("Écart-type du bruit")
    plt.ylabel("Déviation du centroïde (voxels)")
    plt.title("Impact du bruit sur la position du centroïde")
    plt.grid(True)
    plt.show()

    # 📌 Déviation des longueurs d'axes (X, Y, Z)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Noms des axes
    axis_labels = ["Axe Z", "Axe Y", "Axe X"]

    # Liste de couleurs pour chaque std
    colors = plt.cm.viridis(np.linspace(0, 1, len(std_values)))

    # Définition des positions X pour chaque axe (Z=1, Y=2, X=3)
    x_positions = np.array([1, 2, 3])

    for i, std in enumerate(std_values):
        # Calcul des déviations des longueurs des axes
        data = [np.array(results[std]["axes_lengths"])[:, j] - original_axes[j] for j in range(3)]

        # Ajout du boxplot avec un léger décalage horizontal
        ax.boxplot(data, positions=x_positions + (i - len(std_values) / 2) * 0.1, widths=0.08,
                   patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.7))

    # Mise en forme du graphique
    ax.set_xticks(x_positions)
    ax.set_xticklabels(axis_labels)
    ax.set_xlabel("Axes principaux de l'ellipsoïde")
    ax.set_ylabel("Déviation des longueurs d'axes (voxels)")
    ax.set_title("Impact du bruit sur la taille des axes")

    # Légende des couleurs (STD values)
    handles = [plt.Line2D([0], [0], color=colors[i], lw=4, label=f"STD {std_values[i]}") for i in
               range(len(std_values))]
    ax.legend(handles=handles, title="Écart-type du bruit")

    ax.grid(True)
    plt.show()

    # 📌 Déviation angulaire des axes (X, Y, Z)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Noms des axes
    axis_labels = ["Axe Z", "Axe Y", "Axe X"]

    # Liste de couleurs pour chaque std
    colors = plt.cm.viridis(np.linspace(0, 1, len(std_values)))

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
    ax.set_xlabel("Axes principaux de l'ellipsoïde")
    ax.set_ylabel("Déviation angulaire (°)")
    ax.set_title("Impact du bruit sur l'orientation des axes")

    # Légende des couleurs (STD values)
    handles = [plt.Line2D([0], [0], color=colors[i], lw=4, label=f"STD {std_values[i]}") for i in
               range(len(std_values))]
    ax.legend(handles=handles, title="Écart-type du bruit")

    ax.grid(True)
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
        """points = np.array([
            [154., 525.71071978, 782.95301559],
            [154., 545.43582899, 697.09077553],
            [154., 582.56544631, 617.03003817],
            [154., 618.5347631, 519.56479269],
            [154., 625.49656634, 412.81714288],
            [154., 616.21416201, 359.44331797],
            [154., 591.84785064, 302.58859144],
            [140., 644.06137501, 245.73386491],
            [140., 674.22918908, 307.22979361],
            [140., 684.67189396, 381.48902826],
            [140., 697.43519991, 441.82465641],
            [140., 653.34377934, 527.68689648],
            [140., 616.21416201, 601.94613113],
            [140., 603.45085606, 682.00686849],
            [140., 569.80214036, 781.79271505],
            [109., 338.90233261, 219.04695246],
            [109., 315.69632178, 261.97807249],
            [109., 294.81091204, 295.62678819],
            [109., 253.04009255, 382.6493288],
            [109., 224.03257901, 437.18345425],
            [109., 213.58987414, 468.51156886],
            [109., 203.14716926, 518.40449215],
            [109., 182.26175952, 562.49591272],
            [109., 172.97935519, 621.67124033],
            [109., 163.69695086, 668.08326199],
            [109., 153.25424598, 709.85408148],
            [109., 416.64246889, 140.14651564],
            [109., 478.13839758, 134.34501293],
            [109., 647.54227663, 145.94801835],
            [109., 691.63369721, 162.19222593],
            [53., 427.75942144, 168.32708967],
            [53., 400.41644173, 208.20226841],
            [53., 366.2377171, 257.19177372],
            [53., 330.91970164, 307.32056985],
            [53., 279.65161469, 374.53872829],
            [53., 247.7514717, 429.22468771],
            [53., 205.59771132, 485.04993794]
        ])"""
    """
    points = np.array([
        [180.0, 438.579492, 620.621742],
        [180.0, 475.908248, 537.480422],
        [180.0, 487.785580, 457.732625],
        [180.0, 458.940632, 579.899463],
        [180.0, 480.998533, 505.241951],
        [180.0, 464.030917, 418.707107],
        [161.0, 555.656045, 305.024077],
        [161.0, 587.894516, 374.591304],
        [161.0, 601.468609, 484.880811],
        [161.0, 582.804231, 557.841562],
        [161.0, 545.475475, 630.802312],
        [161.0, 523.417574, 724.124203],
        [161.0, 509.843481, 255.817990],
        [174.0, 516.630527, 366.107496],
        [174.0, 535.294905, 447.552055],
        [174.0, 511.540242, 578.202701],
        [174.0, 464.030917, 663.040783],
        [158.0, 531.901382, 720.730679],
        [158.0, 542.081952, 666.434307],
        [158.0, 577.713946, 595.170318],
        [158.0, 592.984801, 532.390137],
        [158.0, 599.771848, 471.306718],
        [158.0, 598.075086, 396.649206],
        [158.0, 587.894516, 332.172263],
        [158.0, 557.352807, 286.359699],
        [158.0, 523.417574, 243.940658],
        [138.0, 664.248790, 293.146746],
        [138.0, 686.306692, 366.107496],
        [138.0, 701.577546, 440.765008],
        [138.0, 669.339075, 508.635474],
        [138.0, 616.739464, 610.441173],
        [138.0, 599.771848, 685.098685],
        [138.0, 598.075086, 739.395057],
        [138.0, 570.926900, 797.084953],
        [138.0, 531.901382, 837.807232],
        [138.0, 621.829749, 240.547135],
        [156.0, 562.443092, 288.056461],
        [156.0, 589.591278, 347.443118],
        [156.0, 611.649179, 418.707107],
        [156.0, 608.255656, 498.454904],
        [156.0, 594.681563, 581.596225],
        [156.0, 557.352807, 656.253737],
        [156.0, 528.507859, 719.033918],
        [174.0, 404.644259, 746.182104],
        [128.0, 548.868999, 820.839616],
        [128.0, 686.306692, 279.572653],
        [145.0, 328.289985, 267.695321],
        [145.0, 382.586358, 233.760088],
        [145.0, 447.063300, 196.431332],
        [145.0, 509.843481, 174.373431],
        [145.0, 287.567706, 345.746357],
        [145.0, 233.271333, 442.461770],
        [145.0, 207.819909, 523.906329],
        [145.0, 178.974961, 578.202701],
        [145.0, 603.165371, 584.989748],
        [145.0, 584.500993, 649.466690],
        [145.0, 543.778714, 785.207622],
        [145.0, 640.494127, 522.209567],
        [145.0, 659.158505, 388.165398],
        [145.0, 659.158505, 320.294932],
        [145.0, 640.494127, 276.179129],
        [145.0, 616.739464, 243.940658],
        [145.0, 567.533377, 186.250762],
        [47.0, 421.611875, 164.192861],
        [47.0, 382.586358, 225.276280],
        [47.0, 365.618741, 277.875891],
        [47.0, 326.120715, 324.998469],
        [47.0, 302.879178, 372.077480],
        [47.0, 268.314840, 413.197123],
        [47.0, 231.366755, 452.528956],
        [47.0, 206.337407, 482.325798],
        [47.0, 183.095870, 514.506388],
        [47.0, 135.420922, 596.149738],
        [66.0, 587.722209, 48.588768],
        [66.0, 538.259450, 67.062810],
        [66.0, 473.898269, 94.475906],
        [66.0, 432.182690, 133.211801],
        [66.0, 386.295552, 189.229866],
        [66.0, 352.923088, 246.439804],
        [66.0, 318.954687, 292.922878],
        [66.0, 299.288771, 322.719721],
        [66.0, 282.006602, 349.536880],
        [66.0, 253.401633, 405.554944],
        [66.0, 228.372285, 443.098966],
        [66.0, 207.098193, 476.860527],
        [66.0, 181.419057, 517.053958],
        [66.0, 151.273983, 572.878168],
        [66.0, 138.992657, 636.517767],
        [66.0, 112.263200, 713.136573],
        [117.0, 702.008268, 231.802523],
        [117.0, 661.484562, 182.935701],
        [117.0, 596.527445, 143.007932],
        [117.0, 520.556277, 128.536708],
        [117.0, 441.285899, 150.866392],
        [117.0, 366.481457, 194.409276],
        [117.0, 318.472637, 243.534581],
        [117.0, 289.444048, 301.591759],
        [117.0, 253.716553, 379.745653],
        [117.0, 230.270385, 436.686347],
        [117.0, 204.591249, 501.442430],
        [117.0, 169.980239, 654.400765],
        [117.0, 140.951649, 762.699733],
        [128.0, 629.971728, 195.525760],
        [128.0, 513.857371, 156.448813],
        [128.0, 248.134132, 404.308305],
        [128.0, 199.008828, 510.374304],
        [128.0, 693.792705, 403.778658],
        [128.0, 648.918997, 517.688838],
        [128.0, 620.441452, 623.832415],
        [128.0, 579.882525, 738.605551],
        [128.0, 302.873678, 279.513007],
        [138.0, 256.274059, 398.600923],
        [138.0, 341.706694, 231.187476],
        [138.0, 243.329720, 484.033558],
        [138.0, 201.044880, 633.324930],
        [138.0, 418.509770, 190.628548],
        [138.0, 484.094419, 168.191695],
        [138.0, 542.775421, 180.273078],
        [158.0, 315.818016, 367.534510],
        [158.0, 469.424168, 224.283829],
        [158.0, 259.725882, 484.033558],
        [158.0, 208.811484, 621.243547],
        [158.0, 190.689409, 715.305742],
    ])
    
    """
    """points = np.array([[61., 118.79660126, 713.53034242],
           [61., 141.63436633, 647.87176786],
           [61., 155.9079695, 609.80882608],
           [61., 171.13314621, 561.27857531],
           [61., 180.64888165, 543.19867797],
           [61., 190.1646171, 526.07035417],
           [61., 202.53507318, 501.32944201],
           [61., 214.90552925, 479.44325049],
           [61., 226.32441179, 455.65391188],
           [61., 242.50116204, 431.86457327],
           [61., 252.01689749, 415.68782301],
           [61., 294.83769247, 348.12610135],
           [61., 313.86916336, 315.77260084],
           [61., 359.54469349, 244.40458501],
           [61., 337.65850197, 275.80651198],
           [61., 283.41880993, 361.44813098],
           [87., 600.29282202, 83.588656],
           [87., 569.84246859, 79.78236182],
           [87., 525.118512, 87.39495017],
           [87., 496.57130567, 90.24967081],
           [87., 460.41151098, 100.7169798],
           [87., 434.71902528, 118.79687714],
           [87., 402.36552477, 154.00509829],
           [87., 374.76989198, 183.50387816],
           [87., 350.98055337, 214.90580513],
           [87., 328.1427883, 247.25930564],
           [87., 296.74086134, 296.74112995],
           [87., 269.14522855, 345.27138072],
           [87., 250.11375766, 386.18904313],
           [112., 619.32429291, 123.55474486],
           [112., 637.40419025, 137.82834803],
           [112., 664.99982304, 162.56926019],
           [112., 698.3048971, 197.77748133],
           [112., 723.04580925, 223.46996703],
           [112., 173.9878741, 652.62963558],
           [112., 188.26147727, 601.24466418],
           [127., 592.68023366, 186.3585988],
           [127., 634.54946962, 218.71209931],
           [127., 654.53251405, 245.35615855],
           [127., 676.41870557, 275.80651198],
           [127., 698.3048971, 325.28833629],
           [127., 713.53007381, 347.17452781],
           [127., 709.72377963, 393.80163149],
           [127., 718.28794153, 370.01229288],
           [136., 503.23232048, 198.72905488],
           [136., 533.6826739, 190.16489298],
           [136., 582.21291741, 196.82590779],
           [136., 617.42113856, 222.51839349],
           [136., 657.38722742, 261.53290881],
           [136., 683.07971312, 325.28833629],
           [136., 694.49859566, 354.78711617],
           [136., 701.15961047, 388.09219022],
           [136., 699.25646338, 430.91299972],
           [136., 684.98286021, 473.73380922],
           [136., 666.90296287, 514.65147163],
           [136., 643.11362426, 564.13329594],
           [136., 617.42113856, 625.03400279],
           [136., 601.2443883, 656.43592976],
           [150., 527.97322538, 832.47703548],
           [150., 549.8594169, 781.09206408],
           [150., 558.4235788, 731.61023977],
           [150., 565.08459361, 699.25673926],
           [150., 571.74560842, 670.70953292],
           [150., 583.16449822, 630.74341502],
           [150., 592.68023366, 605.05092932],
           [150., 603.14754265, 580.31001716],
           [150., 613.61485164, 557.47225209],
           [150., 619.32429291, 539.39235475],
           [150., 627.88845481, 518.45773677],
           [150., 629.79160189, 499.42626588],
           [150., 633.59789607, 470.87905955],
           [150., 637.40419025, 435.6708384],
           [161., 552.38936649, 319.52182902],
           [161., 573.31748332, 368.35410162],
           [161., 585.87435341, 428.34803653],
           [161., 587.2695612, 530.19820511],
           [161., 569.13185995, 580.4256855],
           [161., 546.80853533, 646.00045156],
           [161., 527.27562629, 717.15604878],
           [161., 592.85039236, 479.97072471],
           [79., 724.08576226, 95.10074495],
           [79., 686.47087079, 74.50211391],
           [79., 639.90005278, 69.12855798],
           [79., 597.80719804, 69.12855798],
           [79., 553.923158, 72.7109286],
           [79., 507.35233999, 82.56244779],
           [79., 466.1550779, 100.47430087],
           [79., 436.60052032, 114.80378334],
           [79., 401.67240681, 156.89663808],
           [79., 364.05751534, 210.63219732]])"""

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

    points = points[np.random.choice(points.shape[0], int(0.8 * points.shape[0]), replace=False)]

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

    # Test de sensibilité
    if mstd is False:
        results = test_sensitivity(points, n_simulations=1000, noise_std=2.0, isotropic=True)
        # Analyse des résultats
        analyze_results(centroid, np.sqrt(1 / eigvals), results)
    else:
        std_values = [0.1, 0.5, 1.0, 2.0, 5.0, 8.0]
        results = test_sensitivity_multiple_std(points, n_simulations=10, std_values=std_values, isotropic=True)
        # Analyse des résultats
        analyze_results_multiple_std(centroid, np.sqrt(1 / eigvals), results)


    # Ajout du point d'origine (centroïde)
    viewer.add_points([centroid], name="Origine", size=5, face_color="blue")
    u, v, w = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]
    draw_axes(viewer, centroid, u, v, w)
    viewer.dims.ndisplay = 3
    napari.run()


if __name__ == "__main__":
    main()
