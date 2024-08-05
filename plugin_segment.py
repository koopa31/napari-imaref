import napari
from magicgui import magicgui
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QFileDialog, QPushButton, QFrame, QCheckBox, QHBoxLayout, \
    QLineEdit
from qtpy.QtCore import Qt
import numpy as np
from scipy.optimize import fsolve
from skimage.io import imread
import os
import csv
from napari.utils.notifications import show_info


# Fonction de calcul pour ajuster un plan aux moindres carrés contraint
def ajustement_moindres_carres_contraint(sommet, points):
    """
    Ajuste un plan aux moindres carrés à un ensemble de points,
    en imposant que le plan passe par le sommet donné.

    Args:
    - sommet (np.array): Coordonnées du sommet du cône (doit être un point par lequel le plan passe).
    - points (np.array): Coordonnées des points pour l'ajustement.

    Returns:
    - normal (np.array): Vecteur normal au plan.
    """
    # Convertir les points en coordonnées relatives par rapport au sommet
    points_centres = points - sommet

    # Calcul de la matrice de covariance des points centrés
    covariance_matrix = np.cov(points_centres, rowvar=False)

    # Les vecteurs propres de la matrice de covariance donnent la direction du plus grand écart
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # La direction du vecteur propre correspondant à la plus petite valeur propre est la normale du plan
    normal = eigenvectors[:, 0]  # Vecteur propre associé à la plus petite valeur propre

    return normal


# Fonction de calcul pour le repère à partir des points
def calcul_repere(points):
    """
    Calcule les coordonnées des vecteurs et de l'origine du
    nouveau repère basé sur le cône de révolution.

    Args:
    points (array): coordonnées des points :
    array([
        [z1,y1,x1], # Sommet du cône
        [z2,y2,x2], # Point pour le plan
        [z3,y3,x3], # Point pour le plan
        [z4,y4,x4], # Point pour le plan
        [z5,y5,x5], # Point pour le plan
        [z6,y6,x6], # Point sur la base
    ])

    Returns:
    - origine (tuple): Coordonnées de l'origine du nouveau repère (centre de la base).
    - vecteur_u (tuple): Coordonnées du vecteur unitaire u (ux, uy, uz).
    - vecteur_v (tuple): Coordonnées du vecteur unitaire v (vx, vy, vz).
    - vecteur_w (tuple): Coordonnées du vecteur unitaire w (wx, wy, wz).
    """
    sommet = points[0]
    points_plan = points[1:5]
    point_base = points[5]

    # Calcul du vecteur normal au plan défini par les points 2, 3, 4, 5
    vecteur_v = ajustement_moindres_carres_contraint(sommet, points_plan)
    vecteur_v = vecteur_v / np.linalg.norm(vecteur_v)

    # Calcul du point symétrique de point_base par rapport au plan
    distance_au_plan = np.dot(point_base - sommet, vecteur_v)
    point_sym = point_base - 2 * distance_au_plan * vecteur_v

    # Calcul du centre de la base (milieu entre point_base et point_sym)
    centre_base = (point_base + point_sym) / 2

    # Calcul du vecteur unitaire y (axe de la hauteur, du centre de la base au sommet)
    vecteur_y = sommet - centre_base
    vecteur_y = vecteur_y / np.linalg.norm(vecteur_y)

    # Calcul du vecteur unitaire u (dans le plan de la base, direction de la base)
    vecteur_u = point_base - centre_base
    vecteur_u = vecteur_u / np.linalg.norm(vecteur_u)

    # Calcul du vecteur unitaire w (perpendiculaire aux deux autres)
    vecteur_w = np.cross(vecteur_u, vecteur_y)
    vecteur_w = vecteur_w / np.linalg.norm(vecteur_w)

    return centre_base, vecteur_u, vecteur_y, vecteur_w, sommet, vecteur_v

# Fonction pour redefinir l'origine en fonction de la resolution et d'une distance prédéfinie
def calcul_origine(sommet, vecteur_v, resolution, distance =75):
    """
    Calcule une origine l'origine en fonction de la resolution et d'une distance prédéfinie

    Args:
    - sommet (array): Coordonnées du sommet du cône.
    - vecteur_v (array): Vecteur directeur de la base NORMALISE
    - resolution (array): array contenant la resolution de l'image (en µm/pixel)
    - distance (float): distance en µm entre le sommet et l'origine (par default 75µm)
    """
    vecteur_resolu = np.dot(vecteur_v,resolution)
    vecteur_resolu = vecteur_resolu / np.linalg.norm(vecteur_resolu)
    origine = sommet + vecteur_resolu*distance

    return origine



# Ajuster l'origine pour une hauteur de cône donnée
def ajuster_origine(centre_base, sommet, vecteur_y, hauteur_physique, dx, dy, dz):
    """
    Ajuste l'origine du cône pour correspondre à une hauteur physique donnée.

    Args:
    - centre_base (array): Coordonnées actuelles de l'origine du cône.
    - sommet (array): Coordonnées du sommet du cône.
    - vecteur_y (array): Vecteur unitaire indiquant la direction de la hauteur du cône.
    - hauteur_physique (float): Hauteur physique désirée du cône.
    - dx, dy, dz (float): Résolutions en unités de longueur par pixel/voxel.

    Returns:
    - nouvelle_origine (array): Coordonnées ajustées de l'origine du cône.
    """
    # Calculer la hauteur en pixels en fonction de la résolution et de la hauteur physique
    # Calculer les contributions des résolutions
    facteur = np.sqrt((vecteur_y[0] * dx) ** 2 + (vecteur_y[1] * dy) ** 2 + (vecteur_y[2] * dz) ** 2)

    hauteur_pixels = hauteur_physique / facteur
    vecteur_y_pixels = vecteur_y * hauteur_pixels / np.linalg.norm(vecteur_y)

    # Ajuster l'origine du cône pour correspondre à la nouvelle hauteur
    nouvelle_origine = sommet - vecteur_y_pixels
    return nouvelle_origine, hauteur_pixels


def projection_orthogonale_sur_plan(A, centre_base, v, w):
    """
    Projette orthogonalement un point A sur un plan défini par un point centre_base et les vecteurs v et w.

    Args:
    - A (np.array): Coordonnées du point à projeter.
    - centre_base (np.array): Coordonnées d'un point appartenant au plan.
    - v (np.array): Premier vecteur directeur du plan.
    - w (np.array): Deuxième vecteur directeur du plan.

    Returns:
    - A_prime (np.array): Coordonnées du point projeté sur le plan.
    """
    # Calculer le vecteur normal au plan
    n = np.cross(v, w)

    # Calculer le vecteur AC
    AC = A - centre_base

    # Calculer la projection de AC sur le vecteur normal
    projection_AC_sur_n = (np.dot(AC, n) / np.dot(n, n)) * n

    # Calculer la projection orthogonale de A sur le plan
    A_prime = A - projection_AC_sur_n

    return A_prime

def trouver_nouvelle_origine(S, P, vecteur_v, vecteur_w, d, z_mi_hauteur):
    # Fonction à résoudre pour trouver alpha et beta
    def equations(vars):
        alpha, beta = vars
        O_prime = P + alpha * vecteur_v + beta * vecteur_w
        eq1 = np.linalg.norm(S - O_prime) - d  # Distance au sommet doit être d
        eq2 = O_prime[0] - z_mi_hauteur  # z(O') doit être z_mi_hauteur
        return [eq1, eq2]

    # Résolution des équations
    alpha_beta_initial_guess = [1, 1]  # On peut initialiser avec [0, 0]
    alpha_opt, beta_opt = fsolve(equations, alpha_beta_initial_guess)

    # Calculer la nouvelle origine
    O_prime_opt = P + alpha_opt * vecteur_v + beta_opt * vecteur_w
    return O_prime_opt, alpha_opt, beta_opt


def ajuster_hauteur(centre_base, sommet, vecteur_u, vecteur_y, vecteur_w, points, hauteur_physique, dx, dy, dz):
    """
    Ajuste l'origine et réoriente les vecteurs u, y, et z pour correspondre à la hauteur en z et la distance au sommet.

    Args:
    - centre_base (array): Coordonnées actuelles du centre de la base du cône.
    - sommet (array): Coordonnées du sommet du cône.
    - vecteur_y (array): Vecteur unitaire indiquant la direction de la hauteur du cône.
    - vecteur_u (array): Vecteur directeur de l'axe u du repère.
    - points (array): Coordonnées des points utilisés pour calculer le repère.
    - dx, dy, dz (float): Résolutions en unités de longueur par pixel/voxel.

    Returns:
    - nouvelle_origine (array): Coordonnées ajustées de l'origine du cône.
    - nouveau_vecteur_y (array): Nouveau vecteur y réorienté.
    - nouveau_vecteur_z (array): Nouveau vecteur z réorienté.
    """
    # Projet des points 2 et 5 sur le plan défini par le sommet et vecteur_y
    point_2_proj = projection_orthogonale_sur_plan(points[1], centre_base, vecteur_y, vecteur_w)
    point_5_proj = projection_orthogonale_sur_plan(points[4], centre_base, vecteur_y, vecteur_w)
    mi_hauteur = (point_2_proj + point_5_proj) / 2
    mi_hauteur_z = int(mi_hauteur[0])

    viewer.add_points(mi_hauteur, name="mi_hauteur")
    viewer.add_points(point_2_proj, name="point_2_proj")
    viewer.add_points(point_5_proj, name="point_5_proj")

    # Trouver la nouvelle origine
    nouvelle_origine, alpha, beta = trouver_nouvelle_origine(sommet, centre_base, vecteur_y, vecteur_w, hauteur_physique,
                                                             mi_hauteur_z)

    # Ajuster les vecteurs
    nouveau_vecteur_y = (sommet - nouvelle_origine) / np.linalg.norm(sommet - nouvelle_origine)
    nouveau_vecteur_w = np.cross(vecteur_u, nouveau_vecteur_y)
    nouveau_vecteur_w /= np.linalg.norm(nouveau_vecteur_w)

    return nouvelle_origine, nouveau_vecteur_y, nouveau_vecteur_w


# Fonction pour transformer les coordonnées d'un point
def transformer_coordonnees(point, origine, u, v, w):
    """
    Transforme les coordonnées d'un point depuis le repère original vers le nouveau repère.

    Args:
    - point (array): Coordonnées du point dans le repère original.
    - origine (array): Coordonnées de l'origine du nouveau repère.
    - u (array): Vecteur directeur de l'axe x du nouveau repère.
    - v (array): Vecteur directeur de l'axe y du nouveau repère.
    - w (array): Vecteur directeur de l'axe z du nouveau repère.

    Returns:
    - new_coords (array): Coordonnées du point dans le nouveau repère.
    """
    # Calculer le vecteur de transformation
    transformation_matrix = np.array([u, v, w]).T
    translation = point - origine

    # Résoudre le système linéaire pour obtenir les coordonnées dans le nouveau repère
    new_coords = np.linalg.solve(transformation_matrix, translation)
    return new_coords


# Fonction pour valider les points et calculer le repère
@magicgui(call_button="Calcul des coordonnées")
def valider_points(viewer: napari.Viewer):
    """
    Valide les points placés et calcule le nouveau repère

    Args:
    - viewer: Instance du visualiseur Napari.
    """
    points_layer = viewer.layers['Points']
    points = points_layer.data
    if len(points) != 6:
        show_info("Vous devez placer six points.")
    else:
        global centre_base, u, v, w, nouvelle_origine
        centre_base, u, v, w, sommet, vecteur_v = calcul_repere(points)
        dx = float(dx_input.text())
        dy = float(dy_input.text())
        dz = float(dz_input.text())
        hauteur_physique = float(hauteur_input.text())

        draw_axes(viewer, centre_base, u, v, w)
        draw_origin(viewer, centre_base)

        # Ajuster l'origine du repère en fonction de la hauteur physique
        nouvelle_origine, hauteur_pixels = ajuster_origine(centre_base, sommet, v, hauteur_physique, dx, dy, dz)
        draw_axes(viewer, nouvelle_origine, u, v, w)
        draw_origin(viewer, nouvelle_origine)

        nouvelle_origine, v, w = ajuster_hauteur(nouvelle_origine, sommet, u, v, w, points, hauteur_pixels, dx, dy, dz)

        save_data_to_csv(image_paths[current_image_index], (nouvelle_origine, u, v, w))
        update_instructions(6)
        if show_plane_checkbox.isChecked():
            draw_plane(viewer, sommet, vecteur_v)
        if show_cone_checkbox.isChecked():
            draw_cone(viewer, sommet, nouvelle_origine, points[5], u, w)
        if show_axes_checkbox.isChecked():
            draw_axes(viewer, nouvelle_origine, u, v, w)
            draw_origin(viewer, nouvelle_origine)
        else:
            next_image(viewer)


# Fonction pour dessiner le plan
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


# Fonction pour mettre à jour les instructions
def update_instructions(num_points):
    """
    Met à jour les instructions en fonction du nombre de points placés.

    Args:
    - num_points (int): Nombre de points actuellement placés.
    """
    if num_points == 0:
        instructions_text.setText("Placez le sommet du cône.")
    elif num_points == 1:
        instructions_text.setText("Placez le premier point pour définir le plan.")
    elif num_points == 2:
        instructions_text.setText("Placez le deuxième point pour définir le plan.")
    elif num_points == 3:
        instructions_text.setText("Placez le troisième point pour définir le plan.")
    elif num_points == 4:
        instructions_text.setText("Placez le quatrième point pour définir le plan.")
    elif num_points == 5:
        instructions_text.setText("Placez un point sur le périmètre de la base.")
    elif num_points == 6:
        instructions_text.setText("Cliquez sur 'Calcul des coordonnées' pour lancer les calculs.")
    else:
        instructions_text.setText("Instructions complètes.")


# Fonction pour suivre les changements dans le calque de points
def on_points_update(event):
    """
    Suivi des mises à jour du calque de points, met à jour les instructions.

    Args:
    - event: Événement de mise à jour du calque de points.
    """
    points_layer = event.source
    num_points = len(points_layer.data)
    update_instructions(num_points)


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


# Fonction pour sauvegarder les données dans un fichier CSV
def save_data_to_csv(image_path, data):
    """
    Sauvegarde les données calculées dans un fichier CSV

    Args:
    - image_path (str): Chemin de l'image en cours de traitement.
    - data (tuple): Données contenant l'origine et les vecteurs u, v, w.
    """
    csv_filename = os.path.splitext(image_path)[0] + '.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Origine', 'u', 'v', 'w'])
        csv_writer.writerow(data)


# Fonction pour charger une image et créer un layer de points
def load_image(viewer, image_path):
    """
    Charge une image et ajoute un calque de points pour la saisie des points.
    :param viewer:
    :param image_path:
    :return:
    """
    image = imread(image_path)
    viewer.add_image(image, name='Image 3D')
    points_layer = viewer.add_points(ndim=3, name='Points')
    points_layer.mode = 'add'
    points_layer.events.data.connect(on_points_update)  # Connecte l'événement de mise à jour
    viewer.dims.ndisplay = 2


# Fonction pour sélectionner un dossier et charger la première image
def select_folder_and_load_image(viewer):
    """
    Ouvre une boîte de dialogue pour sélectionner un dossier contenant les images à traiter.
    :param viewer:
    :return:
    """
    global image_paths, current_image_index
    folder = QFileDialog.getExistingDirectory(None, "Sélectionner Dossier")
    if folder:
        image_paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        current_image_index = 0
        if image_paths:
            viewer.layers.clear()
            load_image(viewer, image_paths[current_image_index])
            update_instructions(0)
            image_index.setText(str(current_image_index + 1))
        else:
            show_info("Aucune image trouvée dans le dossier.")


# Fonction pour charger l'image suivante
def next_image(viewer):
    """
    Charge l'image suivante.
    :param viewer:
    :return:
    """
    global current_image_index, image_paths
    if current_image_index < len(image_paths) - 1:
        current_image_index += 1
        viewer.layers.clear()
        load_image(viewer, image_paths[current_image_index])
        update_instructions(0)
        image_index.setText(str(current_image_index + 1))
    else:
        show_info("Toutes les images ont été traitées.")


# Fonction pour charger l'image précédente
def previous_image(viewer):
    """
    Charge l'image précédente.
    :param viewer:
    :return:
    """
    global current_image_index, image_paths
    if current_image_index > 0:
        current_image_index -= 1
        viewer.layers.clear()
        load_image(viewer, image_paths[current_image_index])
        update_instructions(0)
        image_index.setText(str(current_image_index + 1))
    else:
        show_info("Vous êtes déjà à la première image.")


# Fonction pour passer en mode 2D et ajouter un point pour calculer ses coordonnées
@magicgui(call_button="Calculer les coordonnées dans le nouveau repère")
def calculer_coordonnees(viewer: napari.Viewer):
    """
    Permet de passer en mode 2D et de placer un point pour calculer ses coordonnées dans le nouveau repère.

    Args:
    - viewer: Instance du visualiseur Napari.
    """
    viewer.dims.ndisplay = 2
    show_info("Placez un point pour calculer ses coordonnées dans le nouveau repère.")
    points_layer = viewer.layers['Points']
    points_layer.mode = 'add'
    viewer.layers.selection.active = points_layer
    points_layer.events.data.connect(lambda event: on_new_point_added(event, viewer))


# Fonction pour traiter l'ajout d'un nouveau point et calculer ses coordonnées
def on_new_point_added(event, viewer):
    """
    Gère l'ajout d'un nouveau point et calcule ses coordonnées dans le nouveau repère.

    Args:
    - event: Événement de mise à jour du calque de points.
    - viewer: Instance du visualiseur Napari.
    """
    points_layer = event.source
    if len(points_layer.data) > 6:
        new_point = points_layer.data[-1]
        new_coords = transformer_coordonnees(new_point, nouvelle_origine, u, v, w)
        print(f"Coordonnées dans le nouveau repère: {new_coords}")


# Création des champs de saisie pour la résolution
dx_input = QLineEdit()
dx_input.setPlaceholderText("dx (um/pixel)")
dx_input.setText("0.569")
dy_input = QLineEdit()
dy_input.setPlaceholderText("dy (um/pixel)")
dy_input.setText("0.569")
dz_input = QLineEdit()
dz_input.setPlaceholderText("dz (um/pixel)")
dz_input.setText("0.5")
hauteur_input = QLineEdit()
hauteur_input.setPlaceholderText("Hauteur physique (um)")
hauteur_input.setText("75")


# Fonction pour créer un widget combiné
def create_combined_widget(viewer):
    """
    Crée un widget combiné pour l'interface utilisateur
    :param viewer:
    :return:
    """
    combined_widget = QWidget()
    layout = QVBoxLayout()

    select_folder_button = QPushButton("Sélectionner Dossier")
    select_folder_button.clicked.connect(lambda: select_folder_and_load_image(viewer))
    layout.addWidget(select_folder_button)

    layout.addWidget(QLabel("Résolution:"))
    layout.addWidget(QLabel("dx:"))
    layout.addWidget(dx_input)
    layout.addWidget(QLabel("dy:"))
    layout.addWidget(dy_input)
    layout.addWidget(QLabel("dz:"))
    layout.addWidget(dz_input)
    layout.addWidget(QLabel("Hauteur du cône:"))
    layout.addWidget(hauteur_input)

    valider_button = QPushButton("Calcul des coordonnées")
    valider_button.clicked.connect(lambda: valider_points(viewer))
    layout.addWidget(valider_button)

    calculer_coords_button = QPushButton("Calculer les coordonnées dans le nouveau repère")
    calculer_coords_button.clicked.connect(lambda: calculer_coordonnees(viewer))
    layout.addWidget(calculer_coords_button)

    navigation_layout = QHBoxLayout()
    previous_button = QPushButton("<")
    previous_button.clicked.connect(lambda: previous_image(viewer))
    navigation_layout.addWidget(previous_button)

    global image_index
    image_index = QLineEdit("1")
    image_index.setReadOnly(True)
    image_index.setAlignment(Qt.AlignCenter)
    navigation_layout.addWidget(image_index)

    next_button = QPushButton(">")
    next_button.clicked.connect(lambda: next_image(viewer))
    navigation_layout.addWidget(next_button)

    layout.addLayout(navigation_layout)

    global show_axes_checkbox
    show_axes_checkbox = QCheckBox("Afficher les axes")
    layout.addWidget(show_axes_checkbox)

    global show_cone_checkbox
    show_cone_checkbox = QCheckBox("Afficher le cône")
    layout.addWidget(show_cone_checkbox)

    global show_plane_checkbox
    show_plane_checkbox = QCheckBox("Afficher le plan")
    layout.addWidget(show_plane_checkbox)

    instructions_frame = QFrame()
    instructions_frame.setFrameShape(QFrame.Box)
    instructions_frame.setFrameShadow(QFrame.Raised)
    instructions_layout = QVBoxLayout(instructions_frame)

    instructions_label = QLabel("INSTRUCTIONS:")
    instructions_label.setStyleSheet("font-size: 22px; font-family: Arial;")
    instructions_layout.addWidget(instructions_label)

    global instructions_text
    instructions_text = QLabel("")
    instructions_text.setStyleSheet("font-size: 16px; font-family: Arial;")
    instructions_layout.addWidget(instructions_text)

    layout.addWidget(instructions_frame)

    combined_widget.setLayout(layout)
    update_instructions(0)

    return combined_widget


# Variables globales pour les chemins d'image et l'index actuel
image_paths = []
current_image_index = 0

# Créer le viewer Napari sans charger d'image initialement
viewer = napari.Viewer()

# Ajouter le widget combiné au viewer
combined_widget = create_combined_widget(viewer)
viewer.window.add_dock_widget(combined_widget, area='right')

# Ajout d'une image d'accueil pour la fantaisie
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "Image", "Image_accueil.webp")
im_accueil = imread(image_path)
viewer.add_image(im_accueil)

# Lancer l'application Napari
napari.run()
