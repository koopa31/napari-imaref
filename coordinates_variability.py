import os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import re

# Définir le dossier contenant les fichiers CSV
folder_path = "/home/clement/Images/test_plugin_variabilite"

# Initialiser une liste pour stocker les données
all_points = []

# Lire tous les fichiers CSV dans le dossier
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path)
        # Supposons que les coordonnées dans le nouveau repère sont dans les colonnes 'new_x', 'new_y', 'new_z'
        coord_str = data["Coordonnées point de référence"][0]
        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', coord_str)
        coords = np.array([float(num) for num in numbers])
        all_points.append(coords)

# Convertir la liste en un seul tableau numpy
generated_values = np.vstack(np.array(all_points))


# Analyser la variabilité des coordonnées
def analyser_variabilite(coords):
    coords_array = np.array(coords)
    moyenne = np.mean(coords_array, axis=0)
    ecart_type = np.std(coords_array, axis=0)
    variance = np.var(coords_array, axis=0)

    # Calculer les distances pondérées par les résolutions
    differences = coords_array - moyenne
    distances_microns = np.sqrt((dx * differences[:, 0])**2 + (dy * differences[:, 1])**2 + (dz * differences[:, 2])**2)

    distance_moyenne_microns = np.mean(distances_microns)
    coef_variation = ecart_type / moyenne * 100  # En pourcentage
    return moyenne, ecart_type, variance, distance_moyenne_microns, coef_variation


def visualiser_variabilite(coords, moyenne, ecart_type):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    coords_array = np.array(coords)
    ax.scatter(coords_array[:, 0], coords_array[:, 1], coords_array[:, 2], label='Points de référence')
    ax.scatter(moyenne[0], moyenne[1], moyenne[2], color='r', label='Moyenne', s=100)

    # Tracer l'ellipsoïde de confiance
    cov = np.cov(coords_array, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    radii = np.sqrt(eigvals * chi2.ppf(0.95, df=3))

    # Générer les points pour l'ellipsoïde
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

    # Appliquer la rotation des vecteurs propres
    for i in range(len(x)):
        for j in range(len(x)):
            point = np.array([x[i, j], y[i, j], z[i, j]])
            rotated_point = np.dot(eigvecs, point)
            x[i, j], y[i, j], z[i, j] = rotated_point + moyenne

    ax.plot_wireframe(x, y, z, color='r', alpha=0.3, label='Ellipsoïde de Confiance')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Variabilité des coordonnées des points de référence')
    plt.legend()
    plt.show()

# Analyser la variabilité

dx = 0.569
dy = 0.569
dz = 0.5
moyenne, ecart_type, variance, distance_moyenne, coef_variation = analyser_variabilite(generated_values)

print(f"Moyenne des coordonnées : {moyenne}")
print(f"Écart-type des coordonnées : {ecart_type}")
print(f"Variance des coordonnées : {variance}")
print(f"Distance Euclidienne Moyenne à la Moyenne : {distance_moyenne} µm")
print(f"Coefficient de Variation (%): {coef_variation}")

# Visualiser la variabilité
visualiser_variabilite(generated_values, moyenne, ecart_type)
