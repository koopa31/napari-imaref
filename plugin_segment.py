import napari
from magicgui import magicgui
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
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
    image = imread(image_path)  # Utilisation de imread pour lire l'image
    viewer.add_image(image, name='Image 3D')
    viewer.add_shapes(ndim=3, name='Shapes')
    # Connect to on_shape_update function
    viewer.layers['Shapes'].events.data.connect(on_shapes_update)
    # The layer is created with line drawing mode activated
    viewer.layers['Shapes'].mode = "add_line"

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
    viewer.layers.clear()
    global image_paths, current_image_index
    folder = QFileDialog.getExistingDirectory(None, "Select Folder")
    if folder:
        image_paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', 'TIF'))]
        current_image_index = 0
        if image_paths:
            load_image(viewer, image_paths[current_image_index])
            update_instructions(1)
        else:
            show_info("Aucune image trouvée dans le dossier.")

# Fonction pour sauvegarder les données dans un fichier CSV
def save_data_to_csv(image_path, data):
    csv_filename = os.path.splitext(image_path)[0] + '.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Segment 1 Longueur', 'Segment 2 Longueur'])
        csv_writer.writerow(data)

# Widget pour la validation
@magicgui(call_button="Calcul des coordonnées")
def valider_segments(viewer: napari.Viewer):
    shapes_layer = viewer.layers['Shapes']
    segments = shapes_layer.data
    if len(segments) < 2:
        show_info("Vous devez tracer deux segments.")
    else:
        segment1 = segments[0]
        segment2 = segments[1]
        data = calculer_trucs(segment1, segment2)
        save_data_to_csv(image_paths[current_image_index], data)
        update_instructions(4)  # Indiquer que les instructions sont complètes
        next_image(viewer)

# Fonction pour créer un widget d'instructions
def create_instructions_widget():
    instructions_widget = QWidget()
    layout = QVBoxLayout()

    instructions_label = QLabel("Instructions:")
    global instructions_text
    instructions_text = QLabel("")
    update_instructions(1)  # Initialiser avec la première étape

    layout.addWidget(instructions_label)
    layout.addWidget(instructions_text)
    instructions_widget.setLayout(layout)
    return instructions_widget

# Fonction pour suivre les changements dans le layer de shapes
def on_shapes_update(event):
    shapes_layer = event.source
    if len(shapes_layer.data) == 1:
        update_instructions(2)
    elif len(shapes_layer.data) == 0:
        update_instructions(1)
    elif len(shapes_layer.data) == 2:
        update_instructions(3)

# Variables globales pour les chemins d'image et l'index actuel
image_paths = []
current_image_index = 0

# Créer le viewer Napari sans charger d'image initialement
viewer = napari.Viewer()

# Ajouter le bouton de sélection de dossier
select_folder_button = QPushButton("Sélectionner Dossier")
select_folder_button.clicked.connect(lambda: select_folder_and_load_image(viewer))
viewer.window.add_dock_widget(select_folder_button, area='right')

# Ajouter le widget de validation au viewer
viewer.window.add_dock_widget(valider_segments, area='right')

# Ajouter le widget d'instructions au viewer
instructions_widget = create_instructions_widget()
viewer.window.add_dock_widget(instructions_widget, area='right')

# Lancer l'application Napari
napari.run()
