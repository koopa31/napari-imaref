import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from ipywidgets import interact, interactive, fixed, widgets
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Paramètres de l'ellipsoïde cible
a_true, b_true, c_true = 5.0, 3.0, 2.0  # demi-axes
center_true = np.array([1.0, 0.5, -0.5])  # centre
rotation_angles_true = np.array([0.3, 0.2, 0.1])  # angles de rotation en radians


# Génération de points sur une sphère unitaire
def generate_sphere_points(n_points):
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.column_stack((x, y, z))


# Transformation des points de la sphère à l'ellipsoïde
def transform_to_ellipsoid(points, a, b, c, center, rotation_angles):
    # Mise à l'échelle
    scaled_points = points * np.array([a, b, c])

    # Rotation
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
        [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]
    ])

    Ry = np.array([
        [np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
        [0, 1, 0],
        [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]
    ])

    Rz = np.array([
        [np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
        [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    rotated_points = np.array([R @ p for p in scaled_points])

    # Translation
    translated_points = rotated_points + center

    return translated_points


# Génération de données bruitées
def generate_noisy_data(n_points, noise_level=0.2):
    sphere_points = generate_sphere_points(n_points)
    ellipsoid_points = transform_to_ellipsoid(sphere_points, a_true, b_true, c_true,
                                              center_true, rotation_angles_true)

    # Ajouter du bruit
    noise = np.random.normal(0, noise_level, ellipsoid_points.shape)
    noisy_points = ellipsoid_points + noise

    return noisy_points


# Fonction de perte: somme des distances au carré des points à l'ellipsoïde
def loss_function(params, points):
    a, b, c = params[0:3]
    center = params[3:6]
    rotation_angles = params[6:9]

    # Transformation inverse: de l'espace réel à l'espace de la sphère unitaire
    # Translation inverse
    centered_points = points - center

    # Rotation inverse
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
        [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]
    ])

    Ry = np.array([
        [np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
        [0, 1, 0],
        [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]
    ])

    Rz = np.array([
        [np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
        [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    R_inv = np.linalg.inv(R)
    unrotated_points = np.array([R_inv @ p for p in centered_points])

    # Mise à l'échelle inverse
    unscaled_points = unrotated_points / np.array([a, b, c])

    # Calcul de l'erreur pour chaque point (distance à la sphère unitaire)
    norms = np.linalg.norm(unscaled_points, axis=1)
    error = np.sum((norms - 1.0) ** 2)

    return error


# Descente de gradient
def gradient_descent(points, initial_params, learning_rate=0.01, n_iterations=100):
    params = initial_params.copy()
    loss_history = []
    params_history = []

    for i in range(n_iterations):
        # Calculer la perte
        loss = loss_function(params, points)
        loss_history.append(loss)
        params_history.append(params.copy())

        # Calculer le gradient (approximation numérique)
        grad = np.zeros_like(params)
        eps = 1e-6
        for j in range(len(params)):
            params_plus = params.copy()
            params_plus[j] += eps
            loss_plus = loss_function(params_plus, points)

            grad[j] = (loss_plus - loss) / eps

        # Mettre à jour les paramètres
        params -= learning_rate * grad

        # Contraintes sur les demi-axes (doivent être positifs)
        params[0:3] = np.maximum(params[0:3], 0.1)

    return params, loss_history, params_history


# Générer des données bruitées
n_points = 200
noisy_data = generate_noisy_data(n_points, noise_level=0.3)

# Paramètres initiaux (mauvaise estimation délibérée)
initial_params = np.array([
    4.0, 4.0, 4.0,  # demi-axes
    0.0, 0.0, 0.0,  # centre
    0.0, 0.0, 0.0  # angles de rotation
])

# Exécuter la descente de gradient
n_iterations = 200
final_params, loss_history, params_history = gradient_descent(noisy_data,
                                                              initial_params,
                                                              learning_rate=0.005,
                                                              n_iterations=n_iterations)


# Créer une animation de la descente de gradient
def generate_ellipsoid_mesh(a, b, c, center, rotation_angles, resolution=20):
    # Générer une sphère
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(resolution), np.cos(v))

    # Aplatir pour rotation
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)

    # Appliquer la rotation
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
        [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]
    ])

    Ry = np.array([
        [np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
        [0, 1, 0],
        [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]
    ])

    Rz = np.array([
        [np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
        [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx
    rotated_points = np.array([R @ p for p in points])

    # Appliquer la translation
    rotated_points += center

    # Reformer la grille
    x_rot = rotated_points[:, 0].reshape(resolution, resolution)
    y_rot = rotated_points[:, 1].reshape(resolution, resolution)
    z_rot = rotated_points[:, 2].reshape(resolution, resolution)

    return x_rot, y_rot, z_rot


# Créer la figure pour l'animation
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scatter3d"}, {"type": "xy"}]],
    subplot_titles=("Ellipsoid fitting", "Distance between points and the fitted ellipsoid")
)


# Fonction pour créer une figure à une itération donnée
def create_frame_fig(iteration):
    params = params_history[iteration]
    a, b, c = params[0:3]
    center = params[3:6]
    rotation_angles = params[6:9]

    x_ellipsoid, y_ellipsoid, z_ellipsoid = generate_ellipsoid_mesh(
        a, b, c, center, rotation_angles
    )

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "xy"}]],
        subplot_titles=("Ellipsoid fitting", "Distance between points and the fitted ellipsoid")
    )

    # Points de données
    fig.add_trace(
        go.Scatter3d(
            x=noisy_data[:, 0],
            y=noisy_data[:, 1],
            z=noisy_data[:, 2],
            mode='markers',
            marker=dict(size=2, color='blue'),
            name="Manually placed <br> points to fit"
        ),
        row=1, col=1
    )

    # Surface de l'ellipsoïde
    fig.add_trace(
        go.Surface(
            x=x_ellipsoid,
            y=y_ellipsoid,
            z=z_ellipsoid,
            opacity=0.5,
            colorscale='Viridis',
            showscale=False,
            name='Ellipsoïde ajustée'
        ),
        row=1, col=1
    )

    # Calculer les vecteurs d'axes transformés
    # Longueur des flèches (proportionnelles aux demi-axes)
    arrow_scale = min(a, b, c)  # Pour une taille raisonnable
    arrow_length = arrow_scale * 1.5  # Ajustez ce facteur selon les besoins

    # Matrice de rotation pour transformer les axes unitaires
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rotation_angles[0]), -np.sin(rotation_angles[0])],
        [0, np.sin(rotation_angles[0]), np.cos(rotation_angles[0])]
    ])

    Ry = np.array([
        [np.cos(rotation_angles[1]), 0, np.sin(rotation_angles[1])],
        [0, 1, 0],
        [-np.sin(rotation_angles[1]), 0, np.cos(rotation_angles[1])]
    ])

    Rz = np.array([
        [np.cos(rotation_angles[2]), -np.sin(rotation_angles[2]), 0],
        [np.sin(rotation_angles[2]), np.cos(rotation_angles[2]), 0],
        [0, 0, 1]
    ])

    R = Rz @ Ry @ Rx

    # Vecteurs des axes unitaires
    x_axis = R @ np.array([1, 0, 0]) * arrow_length
    y_axis = R @ np.array([0, 1, 0]) * arrow_length
    z_axis = R @ np.array([0, 0, 1]) * arrow_length

    # Ajouter les flèches des axes X, Y, Z
    # Axe X (rouge)
    fig.add_trace(
        go.Scatter3d(
            x=[center[0], center[0] + x_axis[0]],
            y=[center[1], center[1] + x_axis[1]],
            z=[center[2], center[2] + x_axis[2]],
            mode='lines',
            line=dict(color='red', width=6),
            name='Axe X'
        ),
        row=1, col=1
    )

    # Axe Y (vert)
    fig.add_trace(
        go.Scatter3d(
            x=[center[0], center[0] + y_axis[0]],
            y=[center[1], center[1] + y_axis[1]],
            z=[center[2], center[2] + y_axis[2]],
            mode='lines',
            line=dict(color='green', width=6),
            name='Axe Y'
        ),
        row=1, col=1
    )

    # Axe Z (bleu)
    fig.add_trace(
        go.Scatter3d(
            x=[center[0], center[0] + z_axis[0]],
            y=[center[1], center[1] + z_axis[1]],
            z=[center[2], center[2] + z_axis[2]],
            mode='lines',
            line=dict(color='blue', width=6),
            name='Axe Z'
        ),
        row=1, col=1
    )

    # Graphique de la fonction de perte
    fig.add_trace(
        go.Scatter(
            x=list(range(iteration + 1)),
            y=loss_history[:iteration + 1],
            mode='lines+markers',
            name='Loss'
        ),
        row=1, col=2
    )

    # Mise à jour de la mise en page
    fig.update_layout(
        title=f'Iteration {iteration}/{n_iterations - 1} - Loss: {loss_history[iteration]:.2f}',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        xaxis_title='Iteration',
        yaxis_title='Loss function',
        height=600,
        width=1200
    )

    return fig


import os

# Créer un dossier pour les images
output_dir = "ellipsoid_frames"
os.makedirs(output_dir, exist_ok=True)
print(f"Les images seront enregistrées dans: {output_dir}")

# Générer plus d'étapes pour une animation fluide
n_frames = 100
steps = np.linspace(0, n_iterations - 1, n_frames, dtype=int)

# Enregistrer chaque frame
print("Génération des images...")
for i, step_idx in enumerate(steps):
    fig = create_frame_fig(step_idx)

    # Fixer la position de la caméra pour une animation stable
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
    )

    # Enregistrer l'image
    image_path = os.path.join(output_dir, f"frame_{i:04d}.png")
    fig.write_image(image_path, scale=1.5)  # Haute résolution

    # Afficher la progression
    if (i + 1) % 10 == 0 or i == len(steps) - 1:
        print(f"Progression: {i + 1}/{len(steps)} images générées")

print(f"\nToutes les images ont été enregistrées dans {output_dir}")

# Instructions pour convertir en vidéo avec FFmpeg
print("\nPour convertir ces images en vidéo avec FFmpeg, exécutez:")
print(f"ffmpeg -framerate 15 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p ellipsoid_fitting.mp4")

# Alternative avec Python (si ffmpeg est installé)
try:
    import subprocess

    print("\nTentative de création de vidéo avec FFmpeg via Python...")

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", "15",
        "-i", f"{output_dir}/frame_%04d.png",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "ellipsoid_fitting.mp4"
    ]

    result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        print("Vidéo créée avec succès: ellipsoid_fitting.mp4")
    else:
        print("Échec de la création de la vidéo avec FFmpeg")
except Exception as e:
    print(f"Erreur lors de l'utilisation de FFmpeg: {e}")
    print("Vous pouvez toujours utiliser la commande FFmpeg ci-dessus manuellement")

# Afficher les paramètres finaux vs paramètres réels
print("\nParamètres finaux:")
print(f"Demi-axes: a={final_params[0]:.2f}, b={final_params[1]:.2f}, c={final_params[2]:.2f}")
print(f"Centre: ({final_params[3]:.2f}, {final_params[4]:.2f}, {final_params[5]:.2f})")
print(f"Angles de rotation: ({final_params[6]:.2f}, {final_params[7]:.2f}, {final_params[8]:.2f})")

print("\nParamètres réels:")
print(f"Demi-axes: a={a_true:.2f}, b={b_true:.2f}, c={c_true:.2f}")
print(f"Centre: ({center_true[0]:.2f}, {center_true[1]:.2f}, {center_true[2]:.2f})")
print(
    f"Angles de rotation: ({rotation_angles_true[0]:.2f}, {rotation_angles_true[1]:.2f}, {rotation_angles_true[2]:.2f})")

"""# Créer un dossier temporaire pour stocker les images
import os
import tempfile
from moviepy.editor import ImageSequenceClip

temp_dir = tempfile.mkdtemp()
print(f"Images temporaires stockées dans: {temp_dir}")

# Nombre d'images à générer pour l'animation
n_frames = 100
frame_indices = np.linspace(0, n_iterations - 1, n_frames, dtype=int)
image_files = []

print("Génération des images...")
for i, idx in enumerate(frame_indices):
    fig = create_frame_fig(idx)

    # Fixer la position de la caméra pour éviter les sauts
    fig.update_layout(
        scene=dict(
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
    )

    # Sauvegarder l'image
    image_file = os.path.join(temp_dir, f"frame_{i:04d}.png")
    fig.write_image(image_file, scale=1.5)  # Haute résolution
    image_files.append(image_file)

    # Affichage de la progression
    if (i + 1) % 10 == 0 or i == len(frame_indices) - 1:
        print(f"Progression: {i + 1}/{len(frame_indices)} images générées")

# Créer la vidéo à partir des images
print("Création de la vidéo...")
clip = ImageSequenceClip(image_files, fps=15)
output_file = "ellipsoide_fitting.mp4"
clip.write_videofile(output_file, codec='libx264', audio=False)
print(f"Vidéo sauvegardée sous: {output_file}")

# Afficher les paramètres finaux vs paramètres réels
print("Paramètres finaux:")
print(f"Demi-axes: a={final_params[0]:.2f}, b={final_params[1]:.2f}, c={final_params[2]:.2f}")
print(f"Centre: ({final_params[3]:.2f}, {final_params[4]:.2f}, {final_params[5]:.2f})")
print(f"Angles de rotation: ({final_params[6]:.2f}, {final_params[7]:.2f}, {final_params[8]:.2f})")

print("\nParamètres réels:")
print(f"Demi-axes: a={a_true:.2f}, b={b_true:.2f}, c={c_true:.2f}")
print(f"Centre: ({center_true[0]:.2f}, {center_true[1]:.2f}, {center_true[2]:.2f})")
print(
    f"Angles de rotation: ({rotation_angles_true[0]:.2f}, {rotation_angles_true[1]:.2f}, {rotation_angles_true[2]:.2f})")

# Nettoyer les fichiers temporaires (optionnel)
# import shutil
# shutil.rmtree(temp_dir)
# Créer une animation avec des images clés à différentes itérations
steps = np.linspace(0, n_iterations - 1, 20, dtype=int)
for i in steps:
    fig = create_frame_fig(i)
    fig.show()

# Ou, si vous préférez, créer la figure finale
final_fig = create_frame_fig(n_iterations - 1)
final_fig.show()

# Afficher les paramètres finaux vs paramètres réels
print("Paramètres finaux:")
print(f"Demi-axes: a={final_params[0]:.2f}, b={final_params[1]:.2f}, c={final_params[2]:.2f}")
print(f"Centre: ({final_params[3]:.2f}, {final_params[4]:.2f}, {final_params[5]:.2f})")
print(f"Angles de rotation: ({final_params[6]:.2f}, {final_params[7]:.2f}, {final_params[8]:.2f})")

print("\nParamètres réels:")
print(f"Demi-axes: a={a_true:.2f}, b={b_true:.2f}, c={c_true:.2f}")
print(f"Centre: ({center_true[0]:.2f}, {center_true[1]:.2f}, {center_true[2]:.2f})")
print(
    f"Angles de rotation: ({rotation_angles_true[0]:.2f}, {rotation_angles_true[1]:.2f}, {rotation_angles_true[2]:.2f})")"""