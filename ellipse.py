import numpy as np
import napari


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


def run_douglas_rachford(K, gamma=10, max_iter=100, tol=1e-8):
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
    """Visualisation de l'ellipsoïde dans Napari."""
    # Calculer le facteur d'échelle optimal
    scale_factor = refine_scale_factor(A, centroid, eigvecs, viewer.layers["Points"].data)
    eigvals, _ = np.linalg.eigh(A)
    axes_lengths = np.sqrt(1 / eigvals) * scale_factor  # Appliquer le facteur d'échelle

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    u, v = np.meshgrid(u, v)
    x = axes_lengths[0] * np.cos(u) * np.sin(v)
    y = axes_lengths[1] * np.sin(u) * np.sin(v)
    z = axes_lengths[2] * np.cos(v)

    ellipsoid_points = eigvecs @ np.array([x.ravel(), y.ravel(), z.ravel()])
    x_rot, y_rot, z_rot = ellipsoid_points.reshape(3, x.shape[0], x.shape[1]) + centroid[:, None, None]

    vertices = np.column_stack((x_rot.ravel(), y_rot.ravel(), z_rot.ravel()))
    faces = np.array([[i, i + 1, i + 2] for i in range(vertices.shape[0] - 2)])
    viewer.add_surface((vertices, faces), name="Ellipsoïde")

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


def main():

    """# Points ellipse synthétique
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
    ])"""

    # points vrai embryon
    points = np.array([
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
    ])

    center_of_mass = compute_center_of_mass(points)
    normalized_points, P = normalize_points(points, center_of_mass)
    K = compute_k_matrix(normalized_points)

    q = run_douglas_rachford(K)
    A, b, c = extract_parameters(q)

    A, b, c = denormalize_parameters(A, b, c, center_of_mass, P)

    centroid = calculate_centroid(A, b) - center_of_mass
    eigvals, eigvecs = np.linalg.eigh(A)

    viewer = napari.Viewer()
    viewer.add_points(points, name="Points", size=0.5, face_color="red")
    visualize_ellipsoid(A, centroid, eigvecs, viewer)

    # Ajout du point d'origine (centroïde)
    viewer.add_points([centroid], name="Origine", size=5, face_color="blue")
    u, v, w = eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]
    draw_axes(viewer, centroid, u, v, w)
    viewer.dims.ndisplay = 3
    napari.run()


if __name__ == "__main__":
    main()
