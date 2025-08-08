import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
from matplotlib.colors import LinearSegmentedColormap

# Configuration
plt.rcParams['figure.figsize'] = (16, 12)
np.random.seed(42)

# Paramètres de simulation
image_size = 64  # Taille des images
n_frames = 10  # Nombre d'étapes à visualiser (timepoints)
atlas_sigma = 8  # Paramètre de flou pour l'atlas

# Créer un dossier pour sauvegarder les images
output_dir = "recalage_embryons_sequence"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Fonctions pour générer des structures embryonnaires
def create_structure(center, radius, intensity, image_size):
    """Crée une structure circulaire avec un dégradé d'intensité."""
    x, y = np.ogrid[:image_size, :image_size]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist_from_center <= radius
    structure = np.zeros((image_size, image_size))
    structure[mask] = intensity * (1 - dist_from_center[mask] / radius)
    return structure


def create_embryo(centers, radii, intensities, image_size):
    """Crée un embryon avec plusieurs structures."""
    embryo = np.zeros((image_size, image_size))
    for center, radius, intensity in zip(centers, radii, intensities):
        embryo += create_structure(center, radius, intensity, image_size)
    return np.clip(embryo, 0, 1)


# Création de l'atlas (embryon de référence)
atlas_centers = [
    (image_size // 2 - 10, image_size // 2),  # Structure 1 (gauche)
    (image_size // 2 + 10, image_size // 2),  # Structure 2 (droite)
    (image_size // 2, image_size // 2 - 10)  # Structure 3 (haut)
]
atlas_radii = [10, 10, 8]
atlas_intensities = [1.0, 0.8, 0.6]

atlas = create_embryo(atlas_centers, atlas_radii, atlas_intensities, image_size)
atlas = gaussian_filter(atlas, sigma=atlas_sigma / 4)  # Léger flou pour l'atlas

# Création des embryons avec des variations significatives
# Embryon 1 - structures plus espacées et plus grandes
embryo1_centers = [
    (image_size // 2 - 15, image_size // 2 + 3),  # Structure 1 (gauche décalée)
    (image_size // 2 + 15, image_size // 2 - 2),  # Structure 2 (droite décalée)
    (image_size // 2 - 2, image_size // 2 - 15)  # Structure 3 (haut décalée)
]
embryo1_radii = [12, 11, 9]  # Plus grandes structures
embryo1_intensities = [0.9, 0.7, 0.5]

# Embryon 2 - structures plus rapprochées et plus petites
embryo2_centers = [
    (image_size // 2 - 7, image_size // 2 - 4),  # Structure 1 (gauche décalée)
    (image_size // 2 + 7, image_size // 2 + 3),  # Structure 2 (droite décalée)
    (image_size // 2 + 3, image_size // 2 - 7)  # Structure 3 (haut décalée)
]
embryo2_radii = [8, 9, 7]  # Plus petites structures
embryo2_intensities = [1.1, 0.9, 0.7]  # Plus intenses


# Calculer les différences (distances) pour les paramètres
def calculate_parameter_differences():
    """Calcule les différences entre les paramètres des embryons et de l'atlas"""
    # Pour les centres
    diff_centers1 = [(c1[0] - c2[0], c1[1] - c2[1])
                     for c1, c2 in zip(embryo1_centers, atlas_centers)]
    diff_centers2 = [(c1[0] - c2[0], c1[1] - c2[1])
                     for c1, c2 in zip(embryo2_centers, atlas_centers)]

    # Pour les rayons
    diff_radii1 = [r1 - r2 for r1, r2 in zip(embryo1_radii, atlas_radii)]
    diff_radii2 = [r1 - r2 for r1, r2 in zip(embryo2_radii, atlas_radii)]

    # Pour les intensités
    diff_intensities1 = [i1 - i2 for i1, i2 in zip(embryo1_intensities, atlas_intensities)]
    diff_intensities2 = [i1 - i2 for i1, i2 in zip(embryo2_intensities, atlas_intensities)]

    return (diff_centers1, diff_centers2, diff_radii1, diff_radii2,
            diff_intensities1, diff_intensities2)


# Calculer les différences de paramètres
(diff_centers1, diff_centers2, diff_radii1, diff_radii2,
 diff_intensities1, diff_intensities2) = calculate_parameter_differences()


# Fonction pour interpoler les paramètres entre embryon initial et atlas
def interpolate_parameters(initial_params, atlas_params, diff_params, frame, total_frames):
    """Interpole les paramètres entre l'état initial et l'atlas pour une frame donnée"""
    # Calculer le facteur d'interpolation (de 0 à 1)
    t = frame / (total_frames - 1)

    # Fonction d'interpolation non linéaire (sigmoïde) pour un effet plus naturel
    t_smooth = 1 / (1 + np.exp(-10 * (t - 0.5)))

    # Interpoler les paramètres
    if isinstance(initial_params[0], tuple):  # Pour les centres (tuples de coordonnées)
        interp_params = [(initial_params[i][0] - diff_params[i][0] * t_smooth,
                          initial_params[i][1] - diff_params[i][1] * t_smooth)
                         for i in range(len(initial_params))]
    else:  # Pour les rayons et intensités (valeurs simples)
        interp_params = [initial_params[i] - diff_params[i] * t_smooth
                         for i in range(len(initial_params))]

    return interp_params


# Fonction pour créer un embryon pour une frame donnée
def create_frame_embryo(embryo_centers, embryo_radii, embryo_intensities,
                        diff_centers, diff_radii, diff_intensities,
                        frame, total_frames):
    """Crée un embryon à une étape donnée du processus de recalage"""
    # Interpoler les paramètres
    centers = interpolate_parameters(embryo_centers, atlas_centers, diff_centers, frame, total_frames)
    radii = interpolate_parameters(embryo_radii, atlas_radii, diff_radii, frame, total_frames)
    intensities = interpolate_parameters(embryo_intensities, atlas_intensities, diff_intensities, frame, total_frames)

    # Créer l'embryon avec les paramètres interpolés
    embryo = create_embryo(centers, radii, intensities, image_size)

    # Ajouter un effet de mélange avec l'atlas pour les frames avancées
    blend_factor = (frame / (total_frames - 1)) ** 2  # Effet non linéaire
    blended = (1 - blend_factor) * embryo + blend_factor * atlas

    # Ajouter un peu de bruit pour simuler des artefacts de recalage
    noise_level = 0.03 * (frame / (total_frames - 1))  # Augmente avec le recalage
    noise = np.random.normal(0, noise_level, blended.shape)

    return np.clip(blended + noise, 0, 1)


# Calculer la variance spatiale entre deux embryons
def calculate_spatial_variance(embryo1, embryo2):
    """Calcule la variance pixel par pixel entre deux embryons"""
    # Empiler les deux embryons
    stack = np.stack([embryo1, embryo2])
    # Calculer la variance le long de l'axe 0 (entre les deux embryons)
    variance = np.var(stack, axis=0)
    return variance


# Calculer la moyenne de variance spatiale
def mean_spatial_variance(variance_map):
    """Calcule la moyenne de la carte de variance"""
    return np.mean(variance_map)


# Créer et sauvegarder les images pour chaque timepoint
embryo1_frames = []
embryo2_frames = []
variance_frames = []
variance_values = []

for frame in range(n_frames):
    print(f"Traitement de l'image {frame + 1}/{n_frames}...")

    # Créer les embryons pour cette frame
    embryo1 = create_frame_embryo(embryo1_centers, embryo1_radii, embryo1_intensities,
                                  diff_centers1, diff_radii1, diff_intensities1,
                                  frame, n_frames)

    embryo2 = create_frame_embryo(embryo2_centers, embryo2_radii, embryo2_intensities,
                                  diff_centers2, diff_radii2, diff_intensities2,
                                  frame, n_frames)

    # Calculer la variance entre les deux embryons
    variance = calculate_spatial_variance(embryo1, embryo2)
    variance_value = mean_spatial_variance(variance)

    # Stocker les données
    embryo1_frames.append(embryo1)
    embryo2_frames.append(embryo2)
    variance_frames.append(variance)
    variance_values.append(variance_value)

    # Normaliser les valeurs de variance pour le graphique
    variance_values_norm = np.array(variance_values) / variance_values[0]

    # Créer la figure pour ce timepoint
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(1, 3)

    # Sous-figures pour les embryons et la variance
    ax_embryo1 = fig.add_subplot(gs[0, 0])
    ax_embryo2 = fig.add_subplot(gs[0, 1])
    ax_atlas = fig.add_subplot(gs[0, 2])
    """ax_variance = fig.add_subplot(gs[1, 0:2])
    ax_variance_plot = fig.add_subplot(gs[1, 2])"""

    # Afficher les embryons
    ax_embryo1.imshow(embryo1, cmap='Blues')
    ax_embryo1.set_title('Embryo 1')
    ax_embryo1.axis('off')

    ax_embryo2.imshow(embryo2, cmap='Reds')
    ax_embryo2.set_title('Embryo 2')
    ax_embryo2.axis('off')

    # Afficher l'atlas (reste constant)
    ax_atlas.imshow(atlas, cmap='viridis')
    ax_atlas.set_title('Reference atlas')
    ax_atlas.axis('off')

    """# Afficher la carte de variance
    ax_variance.imshow(variance, cmap='hot')
    ax_variance.set_title('Carte de variance entre embryons')
    ax_variance.axis('off')

    # Graphique pour montrer l'évolution de la variance moyenne
    ax_variance_plot.set_xlim(0, n_frames - 1)
    ax_variance_plot.set_ylim(0, 1.1)
    ax_variance_plot.set_xlabel('Étapes de recalage')
    ax_variance_plot.set_ylabel('Variance inter-individuelle\n(normalisée)')
    ax_variance_plot.set_title('Évolution de la variance')
    ax_variance_plot.grid(True, linestyle='--', alpha=0.7)

    # Tracer la ligne de variance et le point actuel
    x = np.arange(frame + 1)
    y = variance_values_norm[:frame + 1]
    ax_variance_plot.plot(x, y, 'k-', lw=2)
    ax_variance_plot.plot(frame, variance_values_norm[frame], 'ro', ms=8)"""

    # Titre global
    progress_text = f'Registration step: {frame} / {n_frames - 1}'
    """if frame == 0:
        progress_text += ' - État initial (différences préservées)'
    elif frame == n_frames - 1:
        progress_text += ' - État final (différences minimisées)'"""
    fig.suptitle(progress_text, fontsize=16)

    # Ajuster la mise en page
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Sauvegarder l'image
    filename = os.path.join(output_dir, f'recalage_timepoint_{frame:02d}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

# Créer une image composite "avant/après" pour montrer la perte de variabilité
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Première ligne: état initial
axes[0, 0].imshow(embryo1_frames[0], cmap='Blues')
axes[0, 0].set_title('Embryon 1 - Initial')
axes[0, 0].axis('off')

axes[0, 1].imshow(embryo2_frames[0], cmap='Reds')
axes[0, 1].set_title('Embryon 2 - Initial')
axes[0, 1].axis('off')

axes[0, 2].imshow(variance_frames[0], cmap='hot')
axes[0, 2].set_title('Variance initiale entre embryons')
axes[0, 2].axis('off')

# Deuxième ligne: état final
axes[1, 0].imshow(embryo1_frames[-1], cmap='Blues')
axes[1, 0].set_title('Embryon 1 - Après recalage')
axes[1, 0].axis('off')

axes[1, 1].imshow(embryo2_frames[-1], cmap='Reds')
axes[1, 1].set_title('Embryon 2 - Après recalage')
axes[1, 1].axis('off')

axes[1, 2].imshow(variance_frames[-1], cmap='hot')
axes[1, 2].set_title('Variance finale entre embryons')
axes[1, 2].axis('off')

# Titre global
fig.suptitle(
    f'Comparaison avant/après le recalage - Réduction de variance: {100 * (1 - variance_values[-1] / variance_values[0]):.1f}%',
    fontsize=18)

# Ajuster la mise en page
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# Sauvegarder l'image composite
filename = os.path.join(output_dir, 'comparaison_avant_apres.png')
plt.savefig(filename, dpi=200, bbox_inches='tight')

print(f"\nTraitement terminé! {n_frames} images ont été sauvegardées dans le dossier '{output_dir}'.")
print(f"Variance initiale entre embryons: {variance_values[0]:.4f}")
print(f"Variance finale entre embryons: {variance_values[-1]:.4f}")
print(f"Réduction de la variance: {100 * (1 - variance_values[-1] / variance_values[0]):.2f}%")
print(
    "\nVous pouvez maintenant utiliser ces images pour créer un GIF ou une vidéo avec un outil comme ImageMagick ou FFmpeg.")