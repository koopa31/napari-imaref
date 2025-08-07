from napari import Viewer
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QHBoxLayout, QFileDialog,
                            QFrame, QProgressBar, QGroupBox, QSizePolicy)
from qtpy.QtCore import Qt
from napari.utils.notifications import show_info
import numpy as np
import os
import pandas as pd
from skimage.io import imread
import importlib.metadata
import requests
import subprocess
import sys
from qtpy.QtWidgets import QMessageBox, QComboBox
from qtpy.QtGui import QIcon, QPixmap
from packaging import version
from qtpy.QtCore import QTimer

# Importing functions from _functions.py
from ._functions import (compute_center_of_mass, normalize_points, compute_k_matrix, run_douglas_rachford,
                         transformer_coordonnees, denormalize_parameters, save_data_to_csv, draw_axes, draw_origin,
                         extract_parameters, calculate_centroid, orienter_axes, visualize_ellipsoid)


# Gestion Bug Napari sur le dessin dans Shapes
def custom_exception_handler(exc_type, exc_value, exc_traceback):
    """Intercepte les erreurs de Napari liées à _moving_value et les ignore."""
    if exc_type == TypeError and "list indices must be integers or slices, not NoneType" in str(exc_value):
        print("⚠️ Napari error ignored : _moving_value is None (well-known bug)")
    else:
        # Afficher les autres erreurs normalement
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Appliquer le hook globalement
sys.excepthook = custom_exception_handler


def check_for_update(plugin_name="napari-stereotypage", viewer=None):
    try:
        # Get the installed version of the plugin
        current_version = importlib.metadata.version(plugin_name)

        # Fetch the latest version available on PyPI
        # TODO: virer test. quand ça sera déployé
        response = requests.get(f"https://test.pypi.org/pypi/{plugin_name}/json")
        response.raise_for_status()  # Raise an exception for HTTP errors
        latest_version = response.json()["info"]["version"]

        # Compare versions using packaging.version
        if version.parse(current_version) < version.parse(latest_version):
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle(f"Update Available for {plugin_name}")
            msg_box.setText(f"A new version ({latest_version}) of {plugin_name} is available.\n"
                            f"You are currently using version {current_version}.\n"
                            f"Do you want to update now?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            # Display the message box and capture the user's response
            result = msg_box.exec_()

            if result == QMessageBox.Yes:
                # The user chose to update
                # TODO: virer "-i https://test.pypi.org/simple/" quand ça sera déployé
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "-i",  "https://test.pypi.org/simple/", plugin_name])

                # Ask the user if they want to restart Napari
                restart_msg = QMessageBox()
                restart_msg.setIcon(QMessageBox.Question)
                restart_msg.setWindowTitle("Restart Napari")
                restart_msg.setText("The plugin has been updated. Please restart Napari to apply changes")
                restart_msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                restart_result = restart_msg.exec_()

                if restart_result == QMessageBox.Yes:
                    # Close and restart Napari
                    sys.exit()
                else:
                    QMessageBox.information(None, "Update Complete",
                                            f"{plugin_name} has been updated, but you chose not to restart Napari.")
        # Useless to know it is already up to date
        """else:
            QMessageBox.information(None, "Up to Date", f"{plugin_name} is already up to date.")"""

    except requests.exceptions.HTTPError as http_err:
        # Manage specific HTTP errors (e.g. 404)
        QMessageBox.critical(None, "Error", f"HTTP error occurred: {http_err}")

    except requests.exceptions.ConnectionError as conn_err:
        # Manage network-related errors
        QMessageBox.critical(None, "Error", f"Connection error occurred: {conn_err}")

    except Exception as e:
        # Manage any other exceptions
        QMessageBox.critical(None, "Error", f"An unexpected error occurred: {e}")



class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setTitle(title)  # Ne pas afficher le titre dans le QGroupBox
        self.setCheckable(False)

        # Ajouter le bouton flèche
        self.toggle_button = QPushButton(title)
        self.toggle_button.setToolTip("Click to show or hide additional options.")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)  # ✅ Ouvrir par défaut
        self.toggle_button.setFixedSize(20, 25)
        self.toggle_button.setStyleSheet("QPushButton { border: none; }")
        self.toggle_button.clicked.connect(self.toggle)

        # Layout du titre
        title_layout = QHBoxLayout()
        title_layout.addWidget(self.toggle_button)
        title_layout.addStretch()
        title_layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(QVBoxLayout())
        self.layout().addLayout(title_layout)

        # Zone de contenu
        self.content_area = QWidget()
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layout().addWidget(self.content_area)

        # ✅ Dérouler par défaut
        self.content_area.setMaximumHeight(16777215)
        self.update_arrow()  # Mettre la flèche à jour

    def toggle(self):
        # Bascule entre déplié et replié
        if self.toggle_button.isChecked():
            self.content_area.setMaximumHeight(16777215)  # Permettre l'expansion
        else:
            self.content_area.setMaximumHeight(0)
        self.update_arrow()

    def setContentLayout(self, layout):
        # Définir le layout de la zone de contenu
        self.content_area.setLayout(layout)

    def update_arrow(self):
        # Mettre à jour la flèche selon l'état déplié/replié
        if self.toggle_button.isChecked():
            self.toggle_button.setText("▼")  # Flèche vers le bas pour indiquer "déplié"
        else:
            self.toggle_button.setText("▶")



class StereotypingWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer = viewer
        # This function ensures add_path mode activation in Lines layer when switching back from 3D to 2D visualization
        self.viewer.dims.events.ndisplay.connect(self.on_display_change)
        self.initUI()
        self.load_intro_image()

    def initUI(self):
        layout = QVBoxLayout(self)

        """# Section déroulante pour les réglages de résolution
        resolution_section = CollapsibleGroupBox("Resolution Settings")
        resolution_layout = QVBoxLayout()

        resolution_layout.addWidget(QLabel("dx (µm/pixel):"))
        self.dx_input = QLineEdit("0.569")
        resolution_layout.addWidget(self.dx_input)

        resolution_layout.addWidget(QLabel("dy (µm/pixel):"))
        self.dy_input = QLineEdit("0.569")
        resolution_layout.addWidget(self.dy_input)

        resolution_layout.addWidget(QLabel("dz (µm/pixel):"))
        self.dz_input = QLineEdit("0.5")
        resolution_layout.addWidget(self.dz_input)

        resolution_section.setContentLayout(resolution_layout)
        layout.addWidget(resolution_section)"""

        # Section déroulante pour la sélection d'images
        image_selection_section = CollapsibleGroupBox("Image Selection")
        image_selection_layout = QVBoxLayout()

        image_selection_layout.addWidget(QLabel("Sample identifier:"))
        self.Sample_id_imput = QLineEdit("_s")
        image_selection_layout.addWidget(self.Sample_id_imput)

        image_selection_layout.addWidget(QLabel("Number of colors:"))
        self.Color_number_imput = QLineEdit("2")
        image_selection_layout.addWidget(self.Color_number_imput)

        image_selection_layout.addWidget(QLabel("Identifier for blue color:"))
        self.Color_id_blue_input = QLineEdit("CSU-405 Em 440-460")
        # Remplace la valeur par défaut par un placeHolder grisé
        """self.Color_id_blue_input = QLineEdit()
        self.Color_id_blue_input.setPlaceholderText("CSU-405 Em 440-460")"""
        image_selection_layout.addWidget(self.Color_id_blue_input)

        image_selection_layout.addWidget(QLabel("Identifier for green color:"))
        self.Color_id_green_input = QLineEdit("CSU-488 Em 510-540")
        image_selection_layout.addWidget(self.Color_id_green_input)

        image_selection_layout.addWidget(QLabel("Identifier for red color:"))
        self.Color_id_red_input = QLineEdit("CSU-561 Em 593LP")
        image_selection_layout.addWidget(self.Color_id_red_input)

        image_selection_section.setContentLayout(image_selection_layout)
        layout.addWidget(image_selection_section)

        select_folder_button = QPushButton("Select Folder")
        select_folder_button.setToolTip("Open the folder containing the images to process")
        select_folder_button.clicked.connect(self.select_folder_and_load_image)
        layout.addWidget(select_folder_button)

        # Section déroulante pour le calcul
        computation_section = CollapsibleGroupBox("Computation section")
        computation_layout = QVBoxLayout()

        # Ajouter les autres composants de l'interface utilisateur
        valider_button = QPushButton("Calculate Coordinates")
        valider_button.setToolTip("Computation of the reference frame")
        valider_button.clicked.connect(self.calculate_coordinates)
        computation_layout.addWidget(valider_button)

        calculer_coords_button = QPushButton("Calculate Coordinates in New Frame")
        calculer_coords_button.setToolTip("Computation of new points coordinates in the previously computed reference frame")
        calculer_coords_button.clicked.connect(self.calculate_new_coordinates)
        computation_layout.addWidget(calculer_coords_button)

        self.dorsal_points_checkbox = QCheckBox("Add dorsal points to orient Z-vector (optional)")
        self.dorsal_points_checkbox.setToolTip("Add aligned dorsal points to help set Z-vector orientation, then click 'Calculate coordinates' to recompute")
        self.dorsal_points_checkbox.stateChanged.connect(self.toggle_dorsal_points)
        computation_layout.addWidget(self.dorsal_points_checkbox)

        self.invert_x_checkbox = QCheckBox("Invert X-vector direction (optional)")
        self.invert_x_checkbox.setToolTip(
            "Check this box if your microscope produces mirror images, causing the left placode to appear on the right "
            "and vice versa. This will invert the direction of the X-axis vector accordingly without affecting Y and Z.")
        self.invert_x_checkbox.stateChanged.connect(self.invert_x)
        computation_layout.addWidget(self.invert_x_checkbox)

        # On afficher toujours l'ellipsoide et les axes mtn
        """self.show_axes_checkbox = QCheckBox("Show computed orthonormal axes")
        self.show_axes_checkbox.setToolTip("Show the orthonormal frame axes")
        computation_layout.addWidget(self.show_axes_checkbox)

        self.show_ellipsoid_checkbox = QCheckBox("Show the computed ellipsoid")
        self.show_ellipsoid_checkbox.setToolTip("Show a transparent fitted ellipsoid")
        computation_layout.addWidget(self.show_ellipsoid_checkbox)"""

        computation_section.setContentLayout(computation_layout)
        layout.addWidget(computation_section)

        # Layout to switch from an image to another
        navigation_layout = QHBoxLayout()
        previous_button = QPushButton("<")
        previous_button.setToolTip("Previous image")
        previous_button.clicked.connect(self.previous_image)
        navigation_layout.addWidget(previous_button)

        self.image_index = QLineEdit("1")
        self.image_index.setToolTip("Image index")
        self.image_index.setReadOnly(False)  # Rendre le champ éditable
        self.image_index.textChanged.connect(self.on_image_index_changed)  # Connecter le changement de texte
        self.image_index.setAlignment(Qt.AlignCenter)
        navigation_layout.addWidget(self.image_index)

        next_button = QPushButton(">")
        next_button.setToolTip("Next image")
        next_button.clicked.connect(self.next_image)
        navigation_layout.addWidget(next_button)

        # Dropdown menu to select image identifier
        self.image_selector = QComboBox()
        self.image_selector.setToolTip("Select an image by identifier")
        self.image_selector.currentIndexChanged.connect(self.on_image_selector_changed)  # Connect selection change
        navigation_layout.addWidget(self.image_selector)

        layout.addLayout(navigation_layout)

        # Layout which shows the instructions
        instructions_frame = QFrame()
        instructions_frame.setFrameShape(QFrame.Box)
        instructions_frame.setFrameShadow(QFrame.Raised)
        instructions_layout = QVBoxLayout(instructions_frame)

        instructions_label = QLabel("INSTRUCTIONS:")
        instructions_label.setStyleSheet("font-size: 22px; font-family: Arial;")
        instructions_layout.addWidget(instructions_label)

        self.instructions_text = QLabel("")
        self.instructions_text.setStyleSheet("font-size: 16px; font-family: Arial;")
        instructions_layout.addWidget(self.instructions_text)

        layout.addWidget(instructions_frame)

        # Add a label for the progress bar
        self.progress_label = QLabel("Progress of processed images:")
        self.progress_label.setStyleSheet(
            "font-size: 16px; font-family: Arial;")  # Optional style to make the text look nicer
        layout.addWidget(self.progress_label)

        # Ajouter la barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # Plage par défaut, ajustée en fonction du nombre d'images
        layout.addWidget(self.progress_bar)  # Ajouter la barre de progression en bas du layout

        # Vertical spacing to separate the doc from the rest of the layout
        layout.addSpacing(80)

        # Ajout du texte d'aide pour la première utilisation
        first_time_label = QLabel("First time using this tool? <br>Check the documentation for guidance!")
        first_time_label.setStyleSheet("font-size: 18px; font-family: Arial; color: white;")  # Style optionnel
        first_time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(first_time_label)

        # Link to the documentation
        open_doc_button = QPushButton("DOCUMENTATION")
        open_doc_button.setToolTip("Open the online documentation")
        # Check if internet connection is working to set the icon, else do nothing
        try:
            url = "https://bitbucket.org/koopa31/napari_svetlana/raw/c7438ec591fa5e23f03cfb17f4984e9f52571649/src/doc.png"
            response = requests.get(url)
            pixmap = QPixmap()
            pixmap.loadFromData(response.content)
            icon = QIcon(pixmap)
            open_doc_button.setIcon(icon)
        except requests.exceptions.RequestException as e:
            print(f"Failed to load documentation button icon: {e}")
            pass
        open_doc_button.clicked.connect(self.open_doc)
        layout.addWidget(open_doc_button)

        load_demo_button = QPushButton("Try on demo image")
        load_demo_button.setToolTip("Load the documentation image to learn how to use the plugin")
        load_demo_button.clicked.connect(self.load_demo_image)
        layout.addWidget(load_demo_button)

        # Set logo if internet connexion works
        """try:
            logo_url = "https://imaref-doc.readthedocs.io/en/latest/_static/logo.png"
            response = requests.get(logo_url)
            if response.status_code == 200:
                pixmap = QPixmap()
                pixmap.loadFromData(response.content)
                pixmap = pixmap.scaled(int(0.8 * layout.sizeHint().width()), int(0.8 * layout.sizeHint().width()),
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_label = QLabel()
                logo_label.setPixmap(pixmap)
                logo_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(logo_label)
        except requests.exceptions.RequestException as e:
            print(f"Failed to load logo: {e}")
            pass"""


        # Set the layout
        self.setLayout(layout)
        self.update_instructions(0)

        check_for_update(plugin_name="napari-stereotypage", viewer=self.viewer)

    def invert_x(self):
        if hasattr(self, 'u'):
            self.u = -self.u  # inversion du vecteur X
            if "Axe X" in self.viewer.layers:
                self.viewer.layers.remove("Axe X")
            if "Axe Y" in self.viewer.layers:
                self.viewer.layers.remove("Axe Y")
            if "Axe Z" in self.viewer.layers:
                self.viewer.layers.remove("Axe Z")
            draw_axes(self.viewer, self.nouvelle_origine, self.u, self.v, self.w)
            save_data_to_csv(
                self.image_paths[self.current_image_index * self.Color_number],
                (self.nouvelle_origine, self.u, self.v, self.w)
            )

    def toggle_dorsal_points(self, state):
        if state == Qt.Checked:
            self.add_dorsal_points_layer()
        else:
            self.remove_dorsal_points_layer()


    def add_dorsal_points_layer(self):
        dorsal_pts_layer = self.viewer.add_points(name="DorsalPoints", ndim=3)
        dorsal_pts_layer.mode = "add"
        self.viewer.dims.ndisplay = 2
        self.viewer.layers.selection.active = dorsal_pts_layer


    def remove_dorsal_points_layer(self):
        if "DorsalPoints" in self.viewer.layers:
            self.viewer.layers.remove("DorsalPoints")

    def on_display_change(self, event):
        """
        # This function ensures add_path mode activation in Lines layer when switching back from 3D to 2D visualization.
        Otherwise, add_path activation does not work automatically.
        :param event:
        :return:
        """
        # If we switch back to 2D
        if self.viewer.dims.ndisplay == 2:
            if self.viewer.layers["Lines"]:
                # If Lines layer exists, we activate add_path mode
                self.viewer.layers["Lines"].mode = "add_path"
                # Once activated, we set Points or Lines layer active to be able to draw directly
                if self.viewer.layers["Points"]:
                    self.viewer.layers.selection.active = self.viewer.layers["Points"]
                else:
                    self.viewer.layers.selection.active = self.viewer.layers["Lines"]

    def load_intro_image(self):
        """
        Load and display the introductory image when the widget is initialized.
        """
        if len(self.viewer.layers) != 0:
            self.viewer.layers.clear()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "Image", "logo.png")
        if os.path.exists(image_path):
            im_accueil = imread(image_path)
            self.viewer.add_image(im_accueil, name="Intro Image")
        else:
            print(f"Introductory image not found at {image_path}")

    def open_doc(self):
        import webbrowser
        webbrowser.open("https://imaref-doc.readthedocs.io/en/latest/##")

    def select_folder_and_load_image(self):
        '''
        Select folder -
        Look for files containing the Sample_id, and reorganise them according
        to the this Sample_id, regardless of the color_id -
        Load images with the same Sample_id
        '''

        #Get parameters input
        self.Sample_id = str(self.Sample_id_imput.text())
        self.Color_number = int(self.Color_number_imput.text())
        self.Color_id_green = str(self.Color_id_green_input.text())
        self.Color_id_blue = str(self.Color_id_blue_input.text())
        self.Color_id_red = str(self.Color_id_red_input.text())
        self.folder = QFileDialog.getExistingDirectory(None, "Select Folder")

        if self.folder:
            image_list = [os.path.join(self.folder, f) for f in sorted(os.listdir(self.folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            self.image_paths = []
            self.identifier_list = []
            while len(image_list)>0:
                #check for presence of the substring identifying files from the same sample
                image_test = image_list[0]
                if not self.Sample_id in image_test:
                    image_list.remove(image_test)
                else:
                    # identifier : string that is unique to a sample, regardless to the color
                    identifier = image_test.split(self.Sample_id)[1]
                    # remove Color_id if present in identifer
                    if self.Color_id_green in identifier and self.Color_id_green != '': 
                        identifier = identifier.split(self.Color_id_green)[0]
                    if self.Color_id_blue in identifier and self.Color_id_blue != '':
                        identifier = identifier.split(self.Color_id_blue)[0]
                    if self.Color_id_red in identifier and self.Color_id_red != '':
                        identifier = identifier.split(self.Color_id_red)[0]
                    identifier = self.Sample_id + identifier
                    if identifier not in self.identifier_list: 
                        self.identifier_list.append(identifier)
                    for file in image_list:
                        # add image corresponding to the same sample to image_paths
                        if identifier in file:
                            self.image_paths.append(file)
                            image_list.remove(file)
            self.current_image_index = 0
            if self.image_paths:
                self.image_selector.clear()
                self.total_images = len(self.image_paths) // self.Color_number

                # Add unique sample id to the menu bar
                self.image_selector.addItems(self.identifier_list)

                # Load first images
                self.load_image_at_index(0)
            else:
                show_info("No images found in the folder.")

    def load_demo_image(self):
        """
        Télécharge une image 3D multicanal de démonstration depuis Zenodo (si pas déjà présente),
        la sauvegarde dans le cache utilisateur, et la charge dans Napari.
        """
        import zipfile
        import urllib.request
        from pathlib import Path

        # 1. Définir les paramètres de l’image de démonstration
        #TODO: replace by correct image
        demo_url = "https://zenodo.org/record/7999871/files/Neural_tube_3D.zip"
        file_hash = "md5:8b459f9224c8f533c97a60b768d4da0a"  # Optionnel si tu veux valider le hash
        cache_dir = Path.home() / ".cache" / "napari-stereotypage"
        cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = cache_dir / "Neural_tube_3D.zip"
        extract_dir = cache_dir / "Neural_tube_3D"
        image_path = extract_dir / "Images" / "neural_tube_3D.tif"

        # 2. Télécharger et extraire si nécessaire
        if not image_path.exists():
            show_info("Downloading demo image, please wait...")
            urllib.request.urlretrieve(demo_url, zip_path)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)

            show_info("Image saved in: {} ".format(extract_dir))

        # Get parameters input
        self.Sample_id = str(self.Sample_id_imput.text())
        self.Color_number = int(self.Color_number_imput.text())
        self.Color_id_green = str(self.Color_id_green_input.text())
        self.Color_id_blue = str(self.Color_id_blue_input.text())
        self.Color_id_red = str(self.Color_id_red_input.text())
        self.folder = os.path.join(cache_dir, "Neural_tube_3D", "Images")

        if self.folder:
            image_list = [os.path.join(self.folder, f) for f in sorted(os.listdir(self.folder)) if
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            self.image_paths = []
            self.identifier_list = []
            while len(image_list) > 0:
                # check for presence of the substring identifying files from the same sample
                image_test = image_list[0]

                for file in image_list:
                    # add image corresponding to the same sample to image_paths
                    self.image_paths.append(file)
                    image_list.remove(file)
            self.current_image_index = 0
            if self.image_paths:
                self.image_selector.clear()
                self.total_images = len(self.image_paths) // self.Color_number

                # Add unique sample id to the menu bar
                self.image_selector.addItems(self.identifier_list)

                # Load first images
                self.load_image_at_index(0)
            else:
                show_info("No images found in the folder.")


    def load_image(self):
        """
        Check the color of each image from the same sample
        Apply blending (additive), auto-contrast, and the appropriate color.
        """

        # Load images depending on current_image_index
        start_index = self.current_image_index * self.Color_number
        end_index = start_index + self.Color_number

        for i, image_path in enumerate(self.image_paths[start_index:end_index]):
            image = imread(image_path)

            # Add appropriate color based on Color_id
            if self.Color_id_red in image_path:
                layer_name = 'Image red'
                colormap = 'red'
            elif self.Color_id_green in image_path:
                layer_name = 'Image green'
                colormap = 'green'
            elif self.Color_id_blue in image_path:
                layer_name = 'Image blue'
                colormap = 'blue'
            else:
                layer_name = f'Image_{i}'
                colormap = 'gray'

            # Add image to napari.viewer with its paramaters
            self.viewer.add_image(image, name=layer_name)
            self.viewer.layers[layer_name].blending = 'additive'
            self.viewer.layers[layer_name].colormap = colormap
            self.viewer.dims.set_point(0, len(self.viewer.layers[layer_name].data) / 2)
            self.viewer.layers[layer_name].reset_contrast_limits()

        # Ajouter un calque de shapes pour l'annotation en traçant des lignes à main levée
        shapes_layer = self.viewer.add_shapes(ndim=3, name='Lines')
        shapes_layer.mode = 'add_path'
        shapes_layer.mouse_drag_callbacks.append(self.on_shape_mouse_release)

        # Ajouter un calque de points pour l'annotation
        points_layer = self.viewer.add_points(ndim=3, name='Points')
        points_layer.mode = 'add'
        points_layer.events.data.connect(self.on_points_update)

        # Viewer in 2D
        self.viewer.dims.ndisplay = 2

        #Ajoute un repere et une ellipsoide si déja calcules
        csv_name = self.image_paths[start_index]
        csv_name = csv_name.rsplit('.',1)[0] + '.csv'
        csv_name = os.path.split(csv_name)[1]
        if csv_name in os.listdir(self.folder):
            df = pd.read_csv(os.path.join(self.folder,csv_name), header=None)
            origin = list(map(float, df[0][1].strip("[]").split()))
            u, v, w = list(map(float, df[1][1].strip("[]").split())), list(map(float, df[2][1].strip("[]").split())), list(map(float, df[3][1].strip("[]").split()))
            origin, u, v, w = np.array(origin), np.array(u), np.array(v), np.array(w)
            draw_axes(self.viewer, origin, u, v, w)
            draw_origin(self.viewer, origin)

    def update_instructions(self, num_points):
        instructions = [
            "Place the apex (anterior part of the embryo)",
            "Place an embryo's dorsal point.",
            "Place at least 7 more points to fit the ellipsoid.",
            "Place at least 6 more points to fit the ellipsoid.",
            "Place at least 5 more points to fit the ellipsoid.",
            "Place at least 4 more points to fit the ellipsoid.",
            "Place at least 3 more points to fit the ellipsoid.",
            "Place at least 2 more points to fit the ellipsoid.",
            "Place at least 1 more points to fit the ellipsoid.",
            "Click 'Calculate Coordinates' to start the calculations."
        ]
        self.instructions_text.setText(instructions[num_points] if num_points < len(instructions) else "Complete instructions.")

    def on_points_update(self, event):
        points_layer = event.source
        num_points = len(points_layer.data)

        # Mise à jour des instructions
        self.update_instructions(num_points)

        # Attribution des couleurs fixes pour les points déjà placés
        colors = []
        for i in range(num_points):
            if i == 0:  # Premier point (apex)
                colors.append([1, 0, 0, 1])  # Rouge
            elif i == 1:  # Points du plan
                colors.append([0, 1, 0, 1])  # Vert
            else:  # Points dorsaux
                colors.append([0, 0, 1, 1])  # Bleu

        # Appliquer les couleurs fixes aux points déjà placés
        points_layer.face_color = np.array(colors)

        # Définir la couleur du prochain point à ajouter
        if num_points == 0:  # Prochain point : Apex
            points_layer.current_face_color = [1, 0, 0, 1]  # Rouge
        elif num_points == 1:  # Prochain point : Plan
            points_layer.current_face_color = [0, 1, 0, 1]  # Vert
        else:  # Prochain point : Dorsaux
            points_layer.current_face_color = [0, 0, 1, 1]  # Bleu

    def on_shape_mouse_release(self, layer, event):
        """Ajoute les points du tracé seulement lorsque la souris est relâchée"""
        # ✅ Attendre activement le relâchement de la souris
        while event.type != "mouse_release":

            yield  # ✅ Attendre sans bloquer

        # Vérifier que des formes existent
        if len(layer.data) > 0:
            last_traced_points = layer.data[-1]  # Dernière forme ajoutée

            if all(point is not None for point in last_traced_points) and len(last_traced_points) > 1:
                # Ajouter les points au calque "Points"
                points_layer = self.viewer.layers["Points"]
                new_points = np.vstack((points_layer.data, last_traced_points)) if len(
                    points_layer.data) > 0 else last_traced_points
                points_layer.data = new_points  # Mise à jour

            # Supprimer le tracé après ajout des points
            layer.data = []


    def adjust_Z_in_plane(self, O, Y, Z0, points):
        """
        Reorient Z-vector in case it is not well fitted enough by adding points in the dorsal plane to define
        a plane it must be contained in.
        :param O: Origin of the orthonormal frame.
        :param Y: Y vector pointing towards the Anterior point.
        :param Z0: Z vector to be adusted
        :param points: Set of points which define the right direction for Z.
        :return: new tuple for (X, Y, Z) vectors.
        """
        if hasattr(points, 'data'):
            points = points.data

        Y = Y / np.linalg.norm(Y)
        Z0 = Z0 / np.linalg.norm(Z0)

        # On cherche le plan passant par O et contenant Y et une direction w (⊥ Y)
        # Pour ce faire, on cherche le vecteur w (⊥ Y) qui minimise la distance au nuage
        # Méthode : trouver le plan contenant Y et qui a une normale N minimisant les distances

        # Centrer les points
        delta = points - O

        # Générer base complète (Y, B1, B2) : B1, B2 dans plan orthogonal à Y
        # Par défaut, on utilise Z0 pour construire B1 (orthogonal à Y)
        B1 = Z0 - (Z0 @ Y) * Y
        if np.linalg.norm(B1) < 1e-8:
            B1 = np.array([1, 0, 0]) if abs(Y[0]) < 0.9 else np.array([0, 1, 0])
            B1 = B1 - (B1 @ Y) * Y
        B1 /= np.linalg.norm(B1)
        B2 = np.cross(Y, B1)

        # On cherche une direction dans le plan orthogonal à Y :
        # direction = cos(θ) * B1 + sin(θ) * B2

        best_theta = None
        min_total_distance = np.inf
        best_dir = None

        for theta in np.linspace(0, 2 * np.pi, 360):
            w = np.cos(theta) * B1 + np.sin(theta) * B2
            # Le plan est span(Y, w), on peut obtenir sa normale :
            N = np.cross(Y, w)
            N = N / np.linalg.norm(N)

            # Calcul de la distance orthogonale de chaque point au plan
            distances = np.abs(delta @ N)
            total = np.mean(distances)

            if total < min_total_distance:
                min_total_distance = total
                best_theta = theta
                best_dir = w

        # Maintenant on a la meilleure direction w ⇒ plan optimal = span(Y, w)
        # On projette Z0 dans ce plan
        A = np.stack([Y, best_dir], axis=1)
        coeffs = np.linalg.lstsq(A, Z0, rcond=None)[0]
        Z_new = A @ coeffs
        Z_new /= np.linalg.norm(Z_new)

        X_new = np.cross(Z_new, Y)
        X_new /= np.linalg.norm(X_new)

        return X_new, Y, Z_new


    # @magicgui(call_button="Calculate Coordinates")
    def calculate_coordinates(self):
        global csv_filename
        points_layer = self.viewer.layers['Points']
        if "DorsalPoints" in self.viewer.layers:
            dorsal_points = self.viewer.layers['DorsalPoints'].data
        points = points_layer.data
        if len(points) < 9:
            show_info("You must place at least 9 points.")
        else:
            """dx = float(self.dx_input.text())
            dy = float(self.dy_input.text())
            dz = float(self.dz_input.text())"""
            # Select two reference points from the dataset.
            PA = points[0, :] # First point in the dataset (used for orientation reference)
            PD = points[1, :] # Second point in the dataset (used for orientation reference)

            # Compute the center of mass of the given points.
            center_of_mass = compute_center_of_mass(points[2:, :])

            # Normalize the points by translating them to the center of mass.
            normalized_points, P = normalize_points(points[2:, :], center_of_mass)

            # Compute the K matrix, which is used for ellipsoid fitting.
            K = compute_k_matrix(normalized_points)

            # Apply the Douglas-Rachford algorithm to optimize the ellipsoid parameters.
            q = run_douglas_rachford(K)

            # Extract the quadratic form parameters (A, b, c) that define the ellipsoid.
            A, b, c = extract_parameters(q)

            # Transform the parameters back to the original coordinate system (denormalization).
            A, b, c = denormalize_parameters(A, b, c, center_of_mass, P)

            # Compute the centroid (geometric center) of the fitted ellipsoid.
            self.nouvelle_origine = calculate_centroid(A, b) - center_of_mass

            # Compute the eigenvalues and eigenvectors of matrix A (which represents the ellipsoid's shape).
            eigvals, eigvecs = np.linalg.eigh(A)

            # Sort the eigenvalues (smallest to largest) and rearrange eigenvectors accordingly.
            order = np.argsort(eigvals)
            eigvecs = eigvecs[:, order]

            # Orient the principal axes (u, v, w) based on the reference points (PA, PD) and the new centroid.
            reoriented_eigvecs = eigvecs.copy()
            reoriented_eigvecs[:, 0], reoriented_eigvecs[:, 1], reoriented_eigvecs[:, 2] = orienter_axes(eigvecs, PA,
                                                                                                         PD, self.nouvelle_origine)
            self.u, self.v, self.w = (reoriented_eigvecs[:, 0].copy(), reoriented_eigvecs[:, 1].copy(),
                                      reoriented_eigvecs[:, 2].copy())

            if (self.dorsal_points_checkbox.isChecked() and "DorsalPoints" in self.viewer.layers and
                    len(self.viewer.layers["DorsalPoints"].data) > 0):
                self.u, self.v, self.w = self.adjust_Z_in_plane(self.nouvelle_origine, self.v, self.w, dorsal_points)

            # Save the computed ellipsoid parameters (centroid and orientation vectors) to a CSV file.
            csv_filename = save_data_to_csv(self.image_paths[self.current_image_index * self.Color_number], (self.nouvelle_origine,
                                                                                             self.u, self.v,
                                                                                             self.w))
            self.update_instructions(10)
            if "Axe X" in self.viewer.layers:
                self.viewer.layers.remove("Axe X")
            if "Axe Y" in self.viewer.layers:
                self.viewer.layers.remove("Axe Y")
            if "Axe Z" in self.viewer.layers:
                self.viewer.layers.remove("Axe Z")
            if "Origine" in self.viewer.layers:
                self.viewer.layers.remove("Origine")
            #if self.show_axes_checkbox.isChecked():
            draw_axes(self.viewer, self.nouvelle_origine, self.u, self.v, self.w)
            draw_origin(self.viewer, self.nouvelle_origine)
            
            if "Ellipsoide" in self.viewer.layers:
                self.viewer.layers.remove("Ellipsoide")
            #if self.show_ellipsoid_checkbox.isChecked():
            visualize_ellipsoid(A, self.nouvelle_origine, eigvecs, self.viewer)
            
            """if not self.show_axes_checkbox.isChecked() and not self.show_ellipsoid_checkbox.isChecked():
                self.next_image()"""


    def calculate_new_coordinates(self):
        """if "Plan" in self.viewer.layers:
            self.viewer.layers.remove("Plan")"""
        self.viewer.dims.ndisplay = 2
        show_info("Place a point to calculate its coordinates in the new frame.")
        self.viewer.add_points(ndim=3, name='Points to evaluate')
        points_layer = self.viewer.layers['Points to evaluate']
        points_layer.mode = 'add'
        self.viewer.layers.selection.active = points_layer
        # ✅ Attendre un peu avant de lier l'événement pour éviter les conflits
        QTimer.singleShot(10, lambda: points_layer.events.data.connect(self.on_new_point_added))

    def on_new_point_added(self, event):
        """ Cette fonction est appelée après qu'un point a été ajouté. """
        points_layer = event.source

        # ✅ Exécuter après un court délai pour éviter le bug "generator already executing"
        QTimer.singleShot(10, lambda: self.process_new_point(points_layer))

    def process_new_point(self, points_layer):
        """ Traitement du point après un court délai pour éviter les conflits avec Napari. """
        if len(points_layer.data) > 0:
            new_point = points_layer.data[-1]
            new_coords = transformer_coordonnees(new_point, self.nouvelle_origine, self.u, self.v, self.w)

            df = pd.read_csv(csv_filename)
            df[f"Point {len(points_layer.data)} Coordinates"] = [new_coords]
            df.to_csv(csv_filename, index=False)

            print(f"Coordinates in the new frame: {new_coords}")

    def update_ui_for_image_change(self, index):
        """Central method to update UI elements when the image changes."""
        if index < 0 or index >= self.total_images:
            show_info("Invalid image index.")
            return

        self.current_image_index = index  # Calculer l'indice de départ
        self.viewer.layers.clear()  # Nettoyer les calques existants
        self.load_image()  # Charger les images

        # Forcer l'activation du dessin dans le Lines layer
        shapes_layer = self.viewer.layers["Lines"]
        if shapes_layer:
            shapes_layer.mode = "add_path"

        # Mettre à jour les widgets sans déclencher d'événements redondants
        self.image_index.blockSignals(True)
        self.image_selector.blockSignals(True)

        self.image_index.setText(str(index + 1))  # Met à jour l'indice affiché
        self.image_selector.setCurrentIndex(index)  # Met à jour la sélection
        self.progress_bar.setValue(int((index / self.total_images) * 100))  # Barre de progression

        self.image_index.blockSignals(False)
        self.image_selector.blockSignals(False)

        # Réinitialiser les instructions
        self.update_instructions(0)

    def load_image_at_index(self, index):
        """Load the image at the specified index and update UI elements."""
        # Mettre à jour l'index de l'image courante
        self.current_image_index = index
        self.viewer.layers.clear()
        self.load_image()

        # Mettre à jour l'index dans le champ `image_index`
        self.image_index.setText(str(index + 1))

        # Mettre à jour la sélection dans le menu déroulant `image_selector`
        self.image_selector.setCurrentIndex(index)

        # Mettre à jour la barre de progression
        if hasattr(self, 'total_images') and self.total_images > 0:
            progress_percentage = int((index / self.total_images) * 100)
            self.progress_bar.setValue(progress_percentage)

    def on_image_selector_changed(self, index):
        if index >= 0 and hasattr(self, 'total_images') and 0 <= index < self.total_images:
            self.load_image_at_index(index)

    def on_image_index_changed(self):
        """Handle manual changes to the image index field."""
        try:
            new_index = int(self.image_index.text()) - 1  # Convertir en index basé sur 0
            current_group = self.current_image_index // self.Color_number

            # Ne rien faire si l'image actuelle est déjà sélectionnée
            if new_index == current_group:
                return

            if 0 <= new_index < self.total_images:
                self.update_ui_for_image_change(new_index)
            else:
                show_info(f"Please enter a valid index between 1 and {self.total_images}.")
        except ValueError:
            show_info("Please enter a valid number.")

    def next_image(self):
        """Navigate to the next image."""
        current_group = self.current_image_index
        if current_group + 1 < self.total_images:
            self.update_ui_for_image_change(current_group + 1)
        else:
            self.progress_bar.setValue(100)
            show_info("You have reached the last image.")

    def previous_image(self):
        """Navigate to the previous image."""
        current_group = self.current_image_index 
        if current_group - 1 >= 0:
            self.update_ui_for_image_change(current_group - 1)
        else:
            show_info("You are already at the first image.")


def make_stereotyping_widget(viewer: Viewer) -> StereotypingWidget:
    return StereotypingWidget(viewer)
