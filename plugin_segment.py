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


# Fonction de calcul qui sera lancée après la validation des segments
def calcul_repere(layer_shape):
    """
    Calcule les coordonees des vecteurs et de l'origine du
    nouveau repere

    Args:
    layer_shape(array): coordonees de 2 segments :
    array([[za1,ya1,xa1],
    [za2,ya2,xa2]],
    [[zb1,yb1,xb1],
    [zb2,yb2,xb2]]))

    Returns:
    - origine (tuple): Coordonnées de l'origine du nouveau repère (x0, y0, z0).
    - vecteur_u (tuple): Coordonnées du vecteur unitaire u (ux, uy, uz).
    - vecteur_v (tuple): Coordonnées du vecteur unitaire v (vx, vy, vz).
    - vecteur_w (tuple): Coordonnées du vecteur unitaire w (wx, wy, wz).
    """

    segment_A = layer_shape[0]
    segment_B = layer_shape[1]

    # Calcul du segment C et de son milieu (Origine du repère)
    millieu_A = (segment_A[0] + segment_A[1]) / 2
    millieu_B = (segment_B[0] + segment_B[1]) / 2
    segment_C = np.array([millieu_A, millieu_B])
    Origine = (segment_C[0] + segment_C[1]) / 2

    # Calcul du vecteur unitaire w (z)
    w = millieu_B - millieu_A
    w = w / np.linalg.norm(w)

    # Calcul du vecteur unitaire u (x)
    v_prim = segment_A[1] - segment_A[0]
    v_prim = v_prim / np.linalg.norm(v_prim)
    u = np.cross(w, v_prim)
    u = u / np.linalg.norm(u)

    # Calcul du vecteur unitaire v (y)
    v = np.cross(u, w)

    return (Origine, u, v, w)


# Fonction pour mettre à jour les instructions
def update_instructions(step):
    if step == 1:
        instructions_text.setText("Tracez le premier segment.")
    elif step == 2:
        instructions_text.setText("Tracez le deuxième segment.")
    elif step == 3:
        instructions_text.setText("Cliquez sur 'Calcul des coordonnées' pour lancer les calculs.")
    else:
        instructions_text.setText("Instructions complètes.")


# Fonction pour charger une image et créer un layer de shapes
def load_image(viewer, image_path):
    image = imread(image_path)
    viewer.add_image(image, name='Image 3D')
    shapes_layer = viewer.add_shapes(ndim=3, name='Shapes')
    shapes_layer.mode = 'add_line'
    shapes_layer.events.data.connect(on_shapes_update)
    viewer.dims.ndisplay = 2


# Fonction pour gérer le changement d'image après validation
def next_image(viewer):
    global current_image_index, image_paths
    if current_image_index < len(image_paths) - 1:
        current_image_index += 1
        viewer.layers.clear()
        load_image(viewer, image_paths[current_image_index])
        update_instructions(1)
        image_index.setText(str(current_image_index + 1))
    else:
        show_info("Toutes les images ont été traitées.")


def previous_image(viewer):
    global current_image_index, image_paths
    if current_image_index > 0:
        current_image_index -= 1
        viewer.layers.clear()
        load_image(viewer, image_paths[current_image_index])
        update_instructions(1)
        image_index.setText(str(current_image_index + 1))
    else:
        show_info("Vous êtes déjà à la première image.")


# Fonction pour sélectionner un dossier et charger la première image
def select_folder_and_load_image(viewer):
    global image_paths, current_image_index
    folder = QFileDialog.getExistingDirectory(None, "Select Folder")
    if folder:
        image_paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if
                       f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', 'TIF'))]
        current_image_index = 0
        if image_paths:
            viewer.layers.clear()
            load_image(viewer, image_paths[current_image_index])
            update_instructions(1)
            image_index.setText(str(current_image_index + 1))
        else:
            show_info("Aucune image trouvée dans le dossier.")


# Fonction pour sauvegarder les données dans un fichier CSV
def save_data_to_csv(image_path, data):
    csv_filename = os.path.splitext(image_path)[0] + '.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Origine', 'u', 'v', 'w'])
        csv_writer.writerow(data)


def draw_axes(viewer, Origine, u, v, w):
    """
    Dessine les 3 axes d'un repère

    Args:
    - Origine (tuple): Coordonnées de l'origine du nouveau repère (x0, y0, z0).
    - vecteur_u (tuple): Coordonnées du vecteur unitaire u (ux, uy, uz).
    - vecteur_v (tuple): Coordonnées du vecteur unitaire v (vx, vy, vz).
    - vecteur_w (tuple): Coordonnées du vecteur unitaire w (wx, wy, wz).
    """
    # Dessine l'axe x
    segment_u = np.array([Origine, Origine + 50 * u])
    viewer.add_shapes(segment_u, shape_type='line', edge_width=1, edge_color='blue', name='Axe X')

    # Dessine l'axe y
    segment_v = np.array([Origine, Origine + 50 * v])
    viewer.add_shapes(segment_v, shape_type='line', edge_width=1, edge_color='red', name='Axe Y')

    # Dessine l'axe z
    segment_w = np.array([Origine, Origine + 50 * w])
    viewer.add_shapes(segment_w, shape_type='line', edge_width=1, edge_color='green', name='Axe Z')

    viewer.dims.ndisplay = 3


# Fonction pour valider les segments
@magicgui(call_button="Calcul des coordonnées")
def valider_segments(viewer: napari.Viewer):
    shapes_layer = viewer.layers['Shapes']
    segments = shapes_layer.data
    if len(segments) < 2:
        show_info("Vous devez tracer deux segments.")
    else:
        data = calcul_repere(segments)
        save_data_to_csv(image_paths[current_image_index], data)
        update_instructions(4)
        if show_axes_checkbox.isChecked():
            draw_axes(viewer, *data)
        else:
            next_image(viewer)


# Fonction pour suivre les changements dans le layer de shapes
def on_shapes_update(event):
    shapes_layer = event.source
    if len(shapes_layer.data) == 1:
        update_instructions(2)
    elif len(shapes_layer.data) == 0:
        update_instructions(1)
    elif len(shapes_layer.data) == 2:
        update_instructions(3)


# Fonction pour créer un widget combiné
def create_combined_widget(viewer):
    combined_widget = QWidget()
    layout = QVBoxLayout()

    select_folder_button = QPushButton("Sélectionner Dossier")
    select_folder_button.clicked.connect(lambda: select_folder_and_load_image(viewer))
    layout.addWidget(select_folder_button)

    valider_button = QPushButton("Calcul des coordonnées")
    valider_button.clicked.connect(lambda: valider_segments(viewer))
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
    update_instructions(1)

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
im_accueil = imread("Image/Image_accueil.webp")
viewer.add_image(im_accueil)

# Lancer l'application Napari
napari.run()
