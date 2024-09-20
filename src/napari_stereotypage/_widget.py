from napari import Viewer
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QCheckBox, QHBoxLayout, QFileDialog,
                            QFrame)
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
from qtpy.QtWidgets import QMessageBox

# Importing functions from _functions.py
from ._functions import (calcul_repere, ajuster_origine, ajuster_hauteur, transformer_coordonnees,
                         calculer_distance_physique, save_data_to_csv, draw_axes, draw_plane, draw_cone, draw_origin)


def check_for_update(plugin_name, viewer: Viewer):
    try:
        # Obtenir la version installée du plugin
        current_version = importlib.metadata.version(plugin_name)

        # Récupérer la dernière version disponible sur PyPI
        response = requests.get(f"https://pypi.org/pypi/{plugin_name}/json")
        latest_version = response.json()["info"]["version"]

        # Comparer les versions
        if current_version != latest_version:
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle(f"Update Available for {plugin_name}")
            msg_box.setText(f"A new version ({latest_version}) of {plugin_name} is available.\n"
                            f"You are currently using version {current_version}.\n"
                            f"Do you want to update now?")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            # Afficher la boîte de dialogue et capturer la réponse de l'utilisateur
            result = msg_box.exec_()

            if result == QMessageBox.Yes:
                # L'utilisateur a choisi de mettre à jour
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", plugin_name])

                # Demander à l'utilisateur s'il souhaite redémarrer Napari
                restart_msg = QMessageBox()
                restart_msg.setIcon(QMessageBox.Question)
                restart_msg.setWindowTitle("Restart Napari")
                restart_msg.setText("The plugin has been updated. To apply the changes, Napari needs to be restarted.\n"
                                    "Would you like to restart now?")
                restart_msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                restart_result = restart_msg.exec_()

                if restart_result == QMessageBox.Yes:
                    # Fermer et relancer Napari
                    restart_napari(viewer)
                else:
                    QMessageBox.information(None, "Update Complete",
                                            f"{plugin_name} has been updated, but you chose not to restart Napari.")
        else:
            QMessageBox.information(None, "Up to Date", f"{plugin_name} is already up to date.")
    except Exception as e:
        if response.status_code == 404:
            QMessageBox.critical(None, "Error", f"An error occurred while checking for updates: the plugin has not been published on pip")
        else:
            QMessageBox.critical(None, "Error", f"An error occurred while checking for updates: {e}")


def restart_napari(viewer: Viewer):
    # Fermer Napari
    viewer.window.close()

    # Relancer Napari
    python_executable = sys.executable
    napari_script = os.path.join(os.path.dirname(python_executable), "napari.exe")  # ou "napari" selon l'installation
    subprocess.Popen([python_executable, "-m", "napari --with napari-stereotypage"])


class StereotypingWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer = viewer
        self.initUI()
        self.load_intro_image()

    def initUI(self):
        layout = QVBoxLayout()

        select_folder_button = QPushButton("Select Folder")
        select_folder_button.clicked.connect(self.select_folder_and_load_image)
        layout.addWidget(select_folder_button)

        layout.addWidget(QLabel("Resolution:"))
        layout.addWidget(QLabel("dx:"))
        self.dx_input = QLineEdit("0.569")
        layout.addWidget(self.dx_input)

        layout.addWidget(QLabel("dy:"))
        self.dy_input = QLineEdit("0.569")
        layout.addWidget(self.dy_input)

        layout.addWidget(QLabel("dz:"))
        self.dz_input = QLineEdit("0.5")
        layout.addWidget(self.dz_input)

        layout.addWidget(QLabel("Cone Height:"))
        self.hauteur_input = QLineEdit("75")
        layout.addWidget(self.hauteur_input)

        valider_button = QPushButton("Calculate Coordinates")
        valider_button.clicked.connect(self.calculate_coordinates)
        layout.addWidget(valider_button)

        calculer_coords_button = QPushButton("Calculate Coordinates in New Frame")
        calculer_coords_button.clicked.connect(self.calculate_new_coordinates)
        layout.addWidget(calculer_coords_button)

        navigation_layout = QHBoxLayout()
        previous_button = QPushButton("<")
        previous_button.clicked.connect(self.previous_image)
        navigation_layout.addWidget(previous_button)

        self.image_index = QLineEdit("1")
        self.image_index.setReadOnly(True)
        self.image_index.setAlignment(Qt.AlignCenter)
        navigation_layout.addWidget(self.image_index)

        next_button = QPushButton(">")
        next_button.clicked.connect(self.next_image)
        navigation_layout.addWidget(next_button)

        layout.addLayout(navigation_layout)

        self.show_axes_checkbox = QCheckBox("Show Axes")
        layout.addWidget(self.show_axes_checkbox)

        self.show_cone_checkbox = QCheckBox("Show Cone")
        layout.addWidget(self.show_cone_checkbox)

        self.show_plane_checkbox = QCheckBox("Show Plane")
        layout.addWidget(self.show_plane_checkbox)

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

        self.setLayout(layout)
        self.update_instructions(0)

        check_for_update(plugin_name="napari-stereotypage", viewer=self.viewer)


    def load_intro_image(self):
        """
        Load and display the introductory image when the widget is initialized.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, os.pardir, os.pardir, "Image", "Image_accueil.webp")
        if os.path.exists(image_path):
            im_accueil = imread(image_path)
            self.viewer.add_image(im_accueil, name="Intro Image")
        else:
            print(f"Introductory image not found at {image_path}")

    def select_folder_and_load_image(self):
        folder = QFileDialog.getExistingDirectory(None, "Select Folder")
        if folder:
            self.image_paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
            self.current_image_index = 0
            if self.image_paths:
                self.viewer.layers.clear()
                self.load_image(self.image_paths[self.current_image_index])
                self.update_instructions(0)
                self.image_index.setText(str(self.current_image_index + 1))
            else:
                show_info("No images found in the folder.")

    def load_image(self, image_path):
        image = imread(image_path)
        self.viewer.add_image(image, name='3D Image')
        points_layer = self.viewer.add_points(ndim=3, name='Points')
        points_layer.mode = 'add'
        points_layer.events.data.connect(self.on_points_update)
        self.viewer.dims.ndisplay = 2

    def update_instructions(self, num_points):
        instructions = [
            "Place the cone's apex.",
            "Place the first point to define the plane.",
            "Place the second point to define the plane.",
            "Place the third point to define the plane.",
            "Place the fourth point to define the plane.",
            "Place the embryo's top point.",
            "Place the embryo's bottom point.",
            "Place a point on the base perimeter.",
            "Click 'Calculate Coordinates' to start the calculations."
        ]
        self.instructions_text.setText(instructions[num_points] if num_points < len(instructions) else "Complete instructions.")

    def on_points_update(self, event):
        points_layer = event.source
        num_points = len(points_layer.data)
        self.update_instructions(num_points)

    #@magicgui(call_button="Calculate Coordinates")
    def calculate_coordinates(self):
        global csv_filename
        points_layer = self.viewer.layers['Points']
        points = points_layer.data
        if len(points) != 8:
            show_info("You must place eight points.")
        else:
            self.centre_base, self.u, self.v, self.w, sommet, vecteur_v, point_haut, point_bas = calcul_repere(points)
            dx = float(self.dx_input.text())
            dy = float(self.dy_input.text())
            dz = float(self.dz_input.text())
            hauteur_physique = float(self.hauteur_input.text())

            self.viewer.add_points(self.centre_base, name="Centre Base")

            self.nouvelle_origine, hauteur_pixels = ajuster_origine(self.centre_base, sommet, self.v, hauteur_physique, dx, dy, dz)
            self.nouvelle_origine, self.v, self.w = ajuster_hauteur(self.nouvelle_origine, sommet, self.u, self.v, self.w, points, hauteur_pixels, dx, dy, dz)

            csv_filename = save_data_to_csv(self.image_paths[self.current_image_index], (self.nouvelle_origine,
                                                                                         self.u, self.v, self.w))
            self.update_instructions(8)
            if self.show_plane_checkbox.isChecked():
                draw_plane(self.viewer, sommet, vecteur_v)
            if self.show_cone_checkbox.isChecked():
                draw_cone(self.viewer, sommet, self.nouvelle_origine, points[5], self.u, self.w)
            if self.show_axes_checkbox.isChecked():
                draw_axes(self.viewer, self.nouvelle_origine, self.u, self.v, self.w)
                draw_origin(self.viewer, self.nouvelle_origine)
            else:
                self.next_image()

            print("Height of the origin in Z: ", self.nouvelle_origine[0])
            print("Distance to apex in µm: ", calculer_distance_physique(self.viewer.layers["Points"].data[0],
                                                                         self.viewer.layers["Origine"].data[0],
                                                                         dx, dy, dz))

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

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.viewer.layers.clear()
            self.load_image(self.image_paths[self.current_image_index])
            self.update_instructions(0)
            self.image_index.setText(str(self.current_image_index + 1))
        else:
            show_info("All images have been processed.")

    def previous_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.viewer.layers.clear()
            self.load_image(self.image_paths[self.current_image_index])
            self.update_instructions(0)
            self.image_index.setText(str(self.current_image_index + 1))
        else:
            show_info("You are already at the first image.")


def make_stereotyping_widget(viewer: Viewer) -> StereotypingWidget:
    return StereotypingWidget(viewer)
