import napari
from magicgui import magicgui
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QFileDialog, QPushButton, QFrame
import numpy as np
from skimage.io import imread
import os
import csv
from napari.utils.notifications import show_info


# Fonction de calcul qui sera lancée après la validation des segments
def calculer_trucs(segment1, segment2):
    longueur1 = np.linalg.norm(segment1[1] - segment1[0])
    longueur2 = np.linalg.norm(segment2[1] - segment2[0])
    print(f"Longueur du segment 1 : {longueur1}")
    print(f"Longueur du segment 2 : {longueur2}")
    return longueur1, longueur2

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
    segment_C = np.array([millieu_A,millieu_B])
    Origine = (segment_C[0] + segment_C[1]) / 2


    # Calcul du vecteur unitaire w (z)
    w = millieu_B - millieu_A
    w = w / np.linalg.norm(w)

    # Calcul du vecteur unitaire u (x)
    v_prim = segment_A[1] - segment_A[0]
    v_prim = v_prim / np.linalg.norm(v_prim)
    u = np.cross(w,v_prim)
    u = u / np.linalg.norm(u)

    # Calcul du vecteur unitaire v (y)
    v = np.cross(u,w)

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


# Fonction pour gérer le changement d'image après validation
def next_image(viewer):
    global current_image_index, image_paths
    if current_image_index < len(image_paths) - 1:
        current_image_index += 1
        viewer.layers.clear()
        load_image(viewer, image_paths[current_image_index])
        update_instructions(1)
    else:
        show_info("Toutes les images ont été traitées.")


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
        else:
            show_info("Aucune image trouvée dans le dossier.")


# Fonction pour sauvegarder les données dans un fichier CSV
def save_data_to_csv(image_path, data):
    csv_filename = os.path.splitext(image_path)[0] + '.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Origine', 'u', 'v', 'w'])
        csv_writer.writerow(data)


# Fonction pour valider les segments
@magicgui(call_button="Calcul des coordonnées")
def valider_segments(viewer: napari.Viewer):
    shapes_layer = viewer.layers['Shapes']
    segments = shapes_layer.data
    if len(segments) < 2:
        show_info("Vous devez tracer deux segments.")
    else:
        # segment1 = segments[0]
        # segment2 = segments[1]
        # data = calculer_trucs(segment1, segment2)
        data = calcul_repere(segments)
        save_data_to_csv(image_paths[current_image_index], data)
        update_instructions(4)
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

# Lancer l'application Napari
napari.run()
