from napari import Viewer
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QHBoxLayout, QFileDialog,
                            QFrame, QProgressBar, QGroupBox, QSizePolicy)
from qtpy.QtCore import Qt
from magicgui import magicgui
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

# Importing functions from _functions.py
from ._functions import (calcul_repere, ajuster_origine, ajuster_hauteur, ajuster_origine_sur_droite_et_distance, transformer_coordonnees,
                         calculer_distance_physique, save_data_to_csv, draw_axes, draw_plane, draw_cone, draw_origin)


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
        self.toggle_button.setChecked(False)
        self.toggle_button.setFixedSize(20, 25)
        self.toggle_button.setStyleSheet("QPushButton { border: none; }")
        self.update_arrow()
        self.toggle_button.clicked.connect(self.toggle)

        # Ajouter le bouton flèche au titre du QGroupBox
        title_layout = QHBoxLayout()
        title_layout.addWidget(self.toggle_button)
        title_layout.addStretch()
        title_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(QVBoxLayout())
        self.layout().addLayout(title_layout)

        # Créer la zone de contenu pour le QGroupBox
        self.content_area = QWidget()
        self.content_area.setMaximumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layout().addWidget(self.content_area)

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
        self.initUI()
        self.load_intro_image()

    def initUI(self):
        layout = QVBoxLayout(self)

        # Section déroulante pour les réglages de résolution
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

        resolution_layout.addWidget(QLabel("Cone Height (µm):"))
        self.hauteur_input = QLineEdit("75")
        resolution_layout.addWidget(self.hauteur_input)

        resolution_layout.addWidget(QLabel("Origin Depth (µm):"))
        self.origin_depth_input = QLineEdit("20")
        resolution_layout.addWidget(self.origin_depth_input)

        resolution_section.setContentLayout(resolution_layout)
        layout.addWidget(resolution_section)

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

        self.show_axes_checkbox = QCheckBox("Show Axes")
        self.show_axes_checkbox.setToolTip("Show the orthonormal frame axes")
        computation_layout.addWidget(self.show_axes_checkbox)

        self.show_cone_checkbox = QCheckBox("Show Cone")
        self.show_cone_checkbox.setToolTip("Show the cone containing the orthonormal frame")
        computation_layout.addWidget(self.show_cone_checkbox)

        self.show_plane_checkbox = QCheckBox("Show Plane")
        self.show_plane_checkbox.setToolTip("Show the embryo symmetry plane")
        computation_layout.addWidget(self.show_plane_checkbox)

        self.method_selection = QComboBox()
        self.method_selection.addItems(
            ["Set origin depth", "Set origin depth and distance to apex"])
        computation_layout.addWidget(QLabel("Calculation Method:"))
        computation_layout.addWidget(self.method_selection)

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

        # Set the layout
        self.setLayout(layout)
        self.update_instructions(0)

        check_for_update(plugin_name="napari-stereotypage", viewer=self.viewer)


    def load_intro_image(self):
        """
        Load and display the introductory image when the widget is initialized.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "Image", "Image_accueil.webp")
        if os.path.exists(image_path):
            im_accueil = imread(image_path)
            self.viewer.add_image(im_accueil, name="Intro Image")
        else:
            print(f"Introductory image not found at {image_path}")

    def open_doc(self):
        import webbrowser
        webbrowser.open("https://stereotyping-doc.readthedocs.io/en/latest/index.html#")

    def select_folder_and_load_image(self):
        self.Sample_id = str(self.Sample_id_imput.text())
        self.Color_number = int(self.Color_number_imput.text())  # Convert Color_number_imput to integer
        folder = QFileDialog.getExistingDirectory(None, "Select Folder")
        if folder:
            image_list = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            self.image_paths = []
            while len(image_list)>0:
                #check the substring identifying files from the same sample
                image_test = image_list[0]
                if not self.Sample_id in image_test:
                    image_list.remove(image_test)
                else:
                    identifier = image_test.split(self.Sample_id)
                    identifier = self.Sample_id + identifier[1]
                    for file in image_list:
                        # reorder image list corresponding to sample files
                        if file.endswith(identifier):
                            self.image_paths.append(file)
                            image_list.remove(file)
            self.current_image_index = 0

            if self.image_paths:
                self.image_selector.clear()
                self.total_images = len(self.image_paths) // self.Color_number

                # Ajouter des identifiants dans le menu déroulant en fonction de Sample_id
                unique_identifiers = [f"{self.Sample_id}{i}" for i in range(1, self.total_images + 1)]
                self.image_selector.addItems(unique_identifiers)

                # Charger la première image
                self.load_image_at_index(0)
            else:
                show_info("No images found in the folder.")

    def load_image(self):
        """
        Check the color of each image from the same sample
        Apply blending (additive), auto-contrast, and the appropriate color.
        """
        # Récupération des paramètres nécessaires
        self.Color_number = int(self.Color_number_imput.text())
        self.Color_id_green = str(self.Color_id_green_input.text())
        self.Color_id_blue = str(self.Color_id_blue_input.text())
        self.Color_id_red = str(self.Color_id_red_input.text())

        # Chargement des images en fonction de current_image_index
        start_index = self.current_image_index
        end_index = start_index + self.Color_number

        for i, image_path in enumerate(self.image_paths[start_index:end_index]):
            image = imread(image_path)

            # Déterminer la couleur basée sur l'identifiant
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

            # Ajouter l'image à Napari avec les paramètres
            self.viewer.add_image(image, name=layer_name)
            self.viewer.layers[layer_name].blending = 'additive'
            self.viewer.layers[layer_name].colormap = colormap
            self.viewer.dims.set_point(0, len(self.viewer.layers[layer_name].data) / 2)
            self.viewer.layers[layer_name].reset_contrast_limits()

        # Ajouter un calque de points pour l'annotation
        points_layer = self.viewer.add_points(ndim=3, name='Points')
        points_layer.mode = 'add'
        points_layer.events.data.connect(self.on_points_update)

        # Affichage en 2D
        self.viewer.dims.ndisplay = 2

    def update_instructions(self, num_points):
        instructions = [
            "Place the cone's apex.",
            "Place the first point to define the plane.",
            "Place the second point to define the plane.",
            "Place the third point to define the plane.",
            "Place the fourth point to define the plane.",
            "Place an embryo's dorsal point.",
            "Place a second embryo's dorsal point.",
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
            elif 1 <= i <= 3:  # Points du plan
                colors.append([0, 0, 1, 1])  # Bleu
            else:  # Points dorsaux
                colors.append([0, 1, 0, 1])  # Vert

        # Appliquer les couleurs fixes aux points déjà placés
        points_layer.face_color = np.array(colors)

        # Définir la couleur du prochain point à ajouter
        if num_points == 0:  # Prochain point : Apex
            points_layer.current_face_color = [1, 0, 0, 1]  # Rouge
        elif 1 <= num_points <= 3:  # Prochain point : Plan
            points_layer.current_face_color = [0, 0, 1, 1]  # Bleu
        else:  # Prochain point : Dorsaux
            points_layer.current_face_color = [0, 1, 0, 1]  # Vert


    #@magicgui(call_button="Calculate Coordinates")
    def calculate_coordinates(self):
        global csv_filename
        points_layer = self.viewer.layers['Points']
        points = points_layer.data
        if len(points) != 7:
            show_info("You must place six points.")
        else:
            dx = float(self.dx_input.text())
            dy = float(self.dy_input.text())
            dz = float(self.dz_input.text())
            distance_apex = float(self.hauteur_input.text())
            origin_depth = float(self.origin_depth_input.text())
            #self.centre_base, self.u, self.v, self.w, sommet, vecteur_v, point_haut, point_bas = calcul_repere(points, origin_depth,distance_apex, dx, dy, dz)
            sommet, vecteur_v, dorsal2_projected, self.nouvelle_origine, self.u, self.v, self.w = calcul_repere(points, origin_depth, distance_apex, dx, dy, dz)
            #sommet, vecteur_v, dorsal2_projected = calcul_repere(points, origin_depth, distance_apex, dx, dy, dz)
            draw_plane(self.viewer, sommet, vecteur_v)
            self.viewer.add_points(dorsal2_projected, name="Point dorsal projeté", face_color="red")
            self.viewer.add_shapes(
                                data=np.array([points[-2], dorsal2_projected]),
                                shape_type="line",
                                edge_color="red",
                                edge_width=2,
                                name="Dorsal to Projected Line",
                            )
            #self.viewer.add_points(self.centre_base, name="Centre Base")

            """self.nouvelle_origine, hauteur_pixels = ajuster_origine(self.centre_base, sommet, self.v, distance_apex, dx, dy, dz)

            if self.method_selection.currentText() == "Constrained by height and apex distance":
                # Existing method
                self.nouvelle_origine, self.v, self.w = ajuster_hauteur(
                    self.nouvelle_origine, sommet, self.u, self.v, self.w, points, hauteur_pixels, dx, dy, dz
                )
            else:
                self.nouvelle_origine, self.v, self.w, M = ajuster_origine_sur_droite_et_distance(
                    sommet, points[5], points[6], self.centre_base, self.u, hauteur_pixels
                )
                self.viewer.add_points(M, name="Milieu", face_color="blue")

            csv_filename = save_data_to_csv(self.image_paths[self.current_image_index-1], (self.nouvelle_origine,
                                                                                         self.u, self.v, self.w))
            self.update_instructions(8)
            print("Height of the origin in Z: ", self.nouvelle_origine[0])
            print("Distance to apex in µm: ", calculer_distance_physique(self.viewer.layers["Points"].data[0],
                                                                         self.nouvelle_origine,
                                                                         dx, dy, dz))"""

            if self.show_plane_checkbox.isChecked():
                draw_plane(self.viewer, sommet, vecteur_v)
            if self.show_cone_checkbox.isChecked():
                draw_cone(self.viewer, sommet, self.nouvelle_origine, points[5], self.u, self.w)
            if self.show_axes_checkbox.isChecked():
                draw_axes(self.viewer, self.nouvelle_origine, self.u, self.v, self.w)
                draw_origin(self.viewer, self.nouvelle_origine)
            else:
                self.next_image()


    def calculate_new_coordinates(self):
        if "Plan" in self.viewer.layers:
            self.viewer.layers.remove("Plan")
        self.viewer.dims.ndisplay = 2
        show_info("Place a point to calculate its coordinates in the new frame.")
        points_layer = self.viewer.layers['Points']
        points_layer.mode = 'add'
        self.viewer.layers.selection.active = points_layer
        points_layer.events.data.connect(self.on_new_point_added)

    def on_new_point_added(self, event):
        points_layer = event.source
        if len(points_layer.data) > 8:
            new_point = points_layer.data[-1]
            new_coords = transformer_coordonnees(new_point, self.nouvelle_origine, self.u, self.v, self.w)

            df = pd.read_csv(csv_filename)
            df[f"Point {len(points_layer.data) - 8} Coordinates"] = [new_coords]
            df.to_csv(csv_filename, index=False)
            print(f"Coordinates in the new frame: {new_coords}")

    def update_ui_for_image_change(self, index):
        """Central method to update UI elements when the image changes."""
        if index < 0 or index >= self.total_images:
            show_info("Invalid image index.")
            return

        self.current_image_index = index * self.Color_number  # Calculer l'indice de départ
        self.viewer.layers.clear()  # Nettoyer les calques existants
        self.load_image()  # Charger les images

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
        self.current_image_index = index * self.Color_number
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
        current_group = self.current_image_index // self.Color_number
        if current_group + 1 < self.total_images:
            self.update_ui_for_image_change(current_group + 1)
        else:
            self.progress_bar.setValue(100)
            show_info("You have reached the last image.")

    def previous_image(self):
        """Navigate to the previous image."""
        current_group = self.current_image_index // self.Color_number
        if current_group - 1 >= 0:
            self.update_ui_for_image_change(current_group - 1)
        else:
            show_info("You are already at the first image.")


def make_stereotyping_widget(viewer: Viewer) -> StereotypingWidget:
    return StereotypingWidget(viewer)
