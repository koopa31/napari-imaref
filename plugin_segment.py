import napari
from magicgui import magicgui
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QFileDialog, QPushButton, QFrame, QCheckBox, QHBoxLayout, \
    QLineEdit
from qtpy.QtCore import Qt
import numpy as np
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
    nouveau repère basé sur le cône de révolution

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
    - points_plan (np.array): Coordonnées des points définissant le plan.
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



# Fonction pour dessiner le plan
def draw_plane(viewer, sommet, vecteur_v):
    """
    Dessine un plan en utilisant un quadrilatère approximant le plan calculé.

    Args:
    - sommet (np.array): Coordonnées du sommet du cône.
    - vecteur_v (np.array): Vecteur normal au plan.
    """
    # Définir un grand quadrilatère autour du sommet pour représenter le plan
    scale_factor = 200  # Ajuster en fonction de l'échelle de vos données
    # Trouver deux vecteurs dans le plan
    arbitrary_vector = np.array([1, 0, 0])
    if np.allclose(np.cross(arbitrary_vector, vecteur_v), 0):
        arbitrary_vector = np.array([0, 1, 0])

    in_plane_vector1 = np.cross(vecteur_v, arbitrary_vector)
    in_plane_vector1 = scale_factor * in_plane_vector1 / np.linalg.norm(in_plane_vector1)
    in_plane_vector2 = np.cross(vecteur_v, in_plane_vector1)
    in_plane_vector2 = scale_factor * in_plane_vector2 / np.linalg.norm(in_plane_vector2)

    # Calcul des coins du quadrilatère
    p1 = sommet + in_plane_vector1 + in_plane_vector2
    p2 = sommet + in_plane_vector1 - in_plane_vector2
    p3 = sommet - in_plane_vector1 - in_plane_vector2
    p4 = sommet - in_plane_vector1 + in_plane_vector2

    # Ajouter le quadrilatère au calque de formes
    viewer.add_shapes(data=[np.array([p1, p2, p3, p4])], shape_type='polygon', edge_width=1, edge_color='cyan',
                      face_color='cyan', opacity=0.3, name='Plan')


# Fonction pour dessiner le cône de révolution
def draw_cone(viewer, sommet, centre_base, point_base, vecteur_u, vecteur_w):
    """
    Dessine une représentation approximative du cône de révolution.

    Args:
    - sommet (array): Coordonnées du sommet du cône.
    - centre_base (array): Coordonnées du centre de la base du cône.
    - point_base (array): Coordonnées d'un point sur la base.
    - vecteur_u (array): Vecteur directeur de la base.
    - vecteur_w (array): Vecteur perpendiculaire pour former le plan de la base.
    """
    # Calcul du rayon de la base à partir du point_base
    rayon_base = np.linalg.norm(point_base - centre_base)

    # Paramètres pour le dessin du cercle de la base
    angles = np.linspace(0, 2 * np.pi, 12)  # 36 points pour approximer le cercle

    # Points de la base
    base_points = [centre_base + rayon_base * (np.cos(angle) * vecteur_u + np.sin(angle) * vecteur_w) for angle in
                   angles]
    base_points = np.array(base_points)

    # Ajouter le cercle de la base
    viewer.add_shapes(data=base_points, shape_type='polygon', edge_width=1, edge_color='yellow', face_color='yellow',
                      opacity=0.3, name='Base du cône')

    # Ajouter les lignes reliant le sommet à la base
    for point in base_points:
        viewer.add_shapes(data=np.array([sommet, point]), shape_type='line', edge_width=1, edge_color='yellow',
                          name='Arête du cône')


# Fonction pour mettre à jour les instructions
def update_instructions(num_points):
    """
    Met à jour les instructions en fonction du nombre de points placés
    :param num_points:
    :return:
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
    Met à jour les instructions en fonction du nombre de points placés.
    :param event:
    :return:
    """
    points_layer = event.source
    num_points = len(points_layer.data)
    update_instructions(num_points)


# Fonction pour valider les points et calculer le repère
@magicgui(call_button="Calcul des coordonnées")
def valider_points(viewer: napari.Viewer):
    """
    Valide les points placés et calcule le nouveau repère
    :param viewer:
    :return:
    """
    points_layer = viewer.layers['Points']
    points = points_layer.data
    if len(points) != 6:
        show_info("Vous devez placer six points.")
    else:
        origine, u, v, w, sommet, vecteur_v = calcul_repere(points)
        save_data_to_csv(image_paths[current_image_index], (origine, u, v, w))
        update_instructions(6)
        if show_plane_checkbox.isChecked():
            draw_plane(viewer, sommet, vecteur_v)
        if show_cone_checkbox.isChecked():
            draw_cone(viewer, points[0], origine, points[5], u, w)
        if show_axes_checkbox.isChecked():
            draw_axes(viewer, origine, u, v, w)
            draw_origin(viewer, origine)
        else:
            next_image(viewer)


# Fonction pour dessiner le repère
def draw_axes(viewer, origine, u, v, w):
    """
    Dessine les 3 axes d'un repère

    Args:
    - Origine (tuple): Coordonnées de l'origine du nouveau repère (x0, y0, z0).
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

    valider_button = QPushButton("Calcul des coordonnées")
    valider_button.clicked.connect(lambda: valider_points(viewer))
    layout.addWidget(valider_button)

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
