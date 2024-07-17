import napari
from magicgui import magicgui
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
import numpy as np

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
        instructions_text.setText("Cliquez sur 'Valider' pour lancer les calculs.")
    else:
        instructions_text.setText("Instructions complètes.")

# Widget pour la validation
@magicgui(call_button="Calcul des coordonnées")
def valider_segments(viewer: napari.Viewer):
    shapes_layer = viewer.layers['Shapes']
    segments = shapes_layer.data
    if len(segments) < 2:
        print("Vous devez tracer deux segments.")
    else:
        segment1 = segments[0]
        segment2 = segments[1]
        calculer_trucs(segment1, segment2)
        update_instructions(4)  # Indiquer que les instructions sont complètes

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


# Charger une image 3D
image_3d = np.random.random((64, 64, 64))

# Créer le viewer Napari et ajouter l'image
viewer = napari.Viewer()

viewer.add_image(image_3d, name='Image 3D')

# Ajouter un layer de shapes
shapes_layer = viewer.add_shapes(name='Shapes')
shapes_layer.events.data.connect(on_shapes_update)

# Ajouter le widget de validation au viewer
viewer.window.add_dock_widget(valider_segments, area='right')

# Ajouter le widget d'instructions au viewer
instructions_widget = create_instructions_widget()
viewer.window.add_dock_widget(instructions_widget, area='right')

"""# Connecter l'événement de pression de touche pour 'Backspace'
viewer.bind_key('Backspace', on_key_press, overwrite=True)"""

# Lancer l'application Napari
napari.run()
