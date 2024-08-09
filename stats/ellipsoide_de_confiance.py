import os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import re

# Définir le dossier contenant les fichiers CSV
folder_path = '/home/clement/Images/test_plugin'
# Définir le niveau de confiance
alpha = 0.95

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


# Calculer la moyenne et la matrice de covariance
mean = np.mean(generated_values, axis=0)
cov = np.cov(generated_values, rowvar=False)

# Calculer les valeurs propres et les vecteurs propres
eigvals, eigvecs = np.linalg.eigh(cov)

# Définir le niveau de confiance
alpha = 0.95

# Calculer les rayons de l'ellipsoïde pour le niveau de confiance donné
radii = np.sqrt(eigvals * chi2.ppf(alpha, df=3))

# Tracer les données
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(generated_values[:, 0], generated_values[:, 1], generated_values[:, 2], alpha=0.6)

# Créer les grilles pour tracer l'ellipsoïde
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = radii[0] * np.outer(np.cos(u), np.sin(v))
y = radii[1] * np.outer(np.sin(u), np.sin(v))
z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

# Créer une matrice de transformation avec les vecteurs propres
transformation_matrix = eigvecs

# Appliquer les transformations pour positionner l'ellipsoïde
ellipsoid_points = np.array([x.flatten(), y.flatten(), z.flatten()])
transformed_ellipsoid_points = transformation_matrix @ ellipsoid_points

# Ajouter la moyenne pour recentrer l'ellipsoïde
x_transformed = transformed_ellipsoid_points[0, :].reshape(x.shape) + mean[0]
y_transformed = transformed_ellipsoid_points[1, :].reshape(y.shape) + mean[1]
z_transformed = transformed_ellipsoid_points[2, :].reshape(z.shape) + mean[2]

# Tracer l'ellipsoïde
ax.plot_wireframe(x_transformed, y_transformed, z_transformed, color='r', alpha=0.3)

# Étiqueter les axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("Ellipsoïde de Confiance à 95%")
plt.show()
