from qtpy.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QFileDialog,
                            QComboBox, QProgressBar, QMessageBox)
from napari.utils.notifications import show_info
from napari import Viewer
import os
import pandas as pd
import numpy as np
import requests
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QPixmap
from xml.etree import ElementTree as ET
from skimage.io import imread
from skimage.measure import regionprops


# Function to load the coordinate system from CSV
def load_coordinate_system(csv_path):
    """
    Loads the new coordinate system's origin and basis vectors from a CSV file.
    The CSV should have three rows: the first for the origin, and the next two for the basis vectors.
    """
    df = pd.read_csv(csv_path, header=None)
    origin = list(map(float, df[0][1].strip("[]").split()))
    u, v, w = (list(map(float, df[1][1].strip("[]").split())), list(map(float, df[2][1].strip("[]").split())),
               list(map(float, df[3][1].strip("[]").split())))

    return origin, u, v, w

# Function to transform points to the new coordinate system
def transform_points(points, origin, u, v, w):
    """
    Transforms a set of points into the new coordinate system.
    """
    transformation_matrix = np.column_stack((u, v, w))
    transformed_points = np.linalg.solve(transformation_matrix, (points - origin).T).T
    return transformed_points


class CoordinateTransformationWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer = viewer
        self.initUI()
        self.load_intro_image()

    def initUI(self):
        """Crée l'interface utilisateur."""
        layout = QVBoxLayout(self)

        # Sélection du dossier
        self.folder_path = QLineEdit()
        self.folder_path.setPlaceholderText("Select a folder containing the data...")
        select_folder_btn = QPushButton("📂 Select a folder")
        select_folder_btn.clicked.connect(self.select_folder)

        # Sélection du type de données
        self.data_type_selection = QComboBox()
        self.data_type_selection.addItems(["User Provided Points", "Segmentation Mask", "Tracking Result"])
        self.data_type_selection.currentIndexChanged.connect(self.update_tracking_format_visibility)

        # Identifiant des fichiers a traiter
        self.file_identifier_input = QLineEdit()

        # Format de tracking (désactivé sauf si Tracking Result est sélectionné)
        self.tracking_format_selection = QComboBox()
        self.tracking_format_selection.addItems(["TrackMate (CSV)", "TrackMate (XML)", "Mastodon/Elephant", "Ultrack"])
        self.tracking_format_selection.setEnabled(False)

        # Bouton de lancement de la transformation
        self.run_button = QPushButton("Transform coordinates")
        self.run_button.clicked.connect(self.run_transformation)

        # Barre de progression
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Ajout des widgets dans le layout
        layout.addWidget(QLabel("Data folder :"))
        layout.addWidget(self.folder_path)
        layout.addWidget(select_folder_btn)
        layout.addWidget(QLabel("Data format :"))
        layout.addWidget(self.data_type_selection)
        layout.addWidget(QLabel("File identifier:"))
        layout.addWidget(self.file_identifier_input)
        layout.addWidget(QLabel("📊 Format de tracking (si applicable) :"))
        layout.addWidget(self.tracking_format_selection)
        layout.addWidget(self.run_button)
        layout.addWidget(self.progress_bar)

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

        self.setLayout(layout)

    def open_doc(self):
        import webbrowser
        webbrowser.open("https://imaref-doc.readthedocs.io/en/latest/##")

    def load_intro_image(self):
        """
        Load and display the introductory image when the widget is initialized.
        """
        if len(self.viewer.layers) != 0:
            self.viewer.layers.clear()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "Image", "Imaref_TransformCoordinates_Instructions_black.png")
        if os.path.exists(image_path):
            im_accueil = imread(image_path)
            self.viewer.add_image(im_accueil, name="Intro Image")
            self.viewer.reset_view()
        else:
            print(f"Introductory image not found at {image_path}")

    def select_folder(self):
        """Ouvre un dialogue pour sélectionner un dossier."""
        folder = QFileDialog.getExistingDirectory(self, "Select a folder")
        if folder:
            self.folder_path.setText(folder)

    def update_tracking_format_visibility(self):
        """Active/Désactive la sélection du format de tracking selon le type de données."""
        if self.data_type_selection.currentText() == "Tracking Result":
            self.tracking_format_selection.setEnabled(True)
        else:
            self.tracking_format_selection.setEnabled(False)

    def run_transformation(self):
        """Lance la transformation des coordonnées selon le type sélectionné."""
        input_folder = self.folder_path.text()
        data_type = self.data_type_selection.currentText()
        tracking_format = self.tracking_format_selection.currentText()
        self.file_identifier = self.file_identifier_input.text()

        if not os.path.exists(input_folder):
            QMessageBox.critical(self, "Error", "The selected folder does not exist.")
            return

        # Ensure the files to analyse exist
        files = [f for f in os.listdir(input_folder) if self.file_identifier in f]
        if not files:
            QMessageBox.critical(self, "Error", "The selected folder does not contain files to analyse.")
            return

        show_info(f"Processing data for {data_type}...")

        if data_type == "User Provided Points":
            self.transform_points_csv(input_folder)
        elif data_type == "Segmentation Mask":
            self.process_segmentation_folder(input_folder)
        else:
            if tracking_format == "TrackMate (CSV)":
                self.transform_trackmate_res(input_folder, "csv")
            elif tracking_format == "TrackMate (XML)":
                self.transform_trackmate_res(input_folder, "xml")
            elif tracking_format == "Mastodon/Elephant":
                self.transform_mastodon_csv(input_folder)
            else:
                self.transform_ultrack_csv(input_folder)

        QMessageBox.information(self, "Successful", "Transformation finished !")
        self.progress_bar.setValue(100)

    # SECTION 1

    def transform_points_csv(self, input_folder):
        """
        Process all CSV files containing point coordinates in the input folder and transform them.
        - Detects corresponding reference CSV files.
        - Updates the original CSV with transformed coordinates.

        Parameters:
        - input_folder: Path to the dataset containing point CSVs and reference coordinate CSVs.
        """
        files = os.listdir(input_folder)
        point_files = [f for f in files if f.endswith(self.file_identifier + ".csv")]

        for index, point_file in enumerate(point_files):
            self.progress_bar.setValue(int(100*index/len(point_files)))
            base_name = point_file.replace((self.file_identifier + ".csv"), "")
            points_csv_path = os.path.join(input_folder, point_file)
            ref_csv_path = os.path.join(input_folder, base_name + "imaref.csv")
            newpoints_csv_path = os.path.join(input_folder, base_name + "newpoints.csv")

            # Ensure the reference CSV exists
            if not os.path.exists(ref_csv_path):
                print(f"⚠️ Missing reference file for {base_name}, skipping.")
                continue

            # Load the reference coordinate system
            origin, u, v, w = load_coordinate_system(ref_csv_path)

            # Load the points CSV
            df = pd.read_csv(points_csv_path)

            # Ensure the required columns exist
            if {"Z", "Y", "X"}.issubset(df.columns):
                points = df[["Z", "Y", "X"]].values
            elif {"axis-0", "axis-1", "axis-2"}.issubset(df.columns):
                points = df[["axis-0", "axis-1", "axis-2"]].values
            else:
                print(f"⚠️ Skipping {point_file}: Missing required columns (Z, Y, X).")
                continue


            # Transform the points
            transformed_points = transform_points(points, origin, u, v, w)
            #put back the coordinates in the [z,y,x] order for consistency with napari
            transformed_points[:,[0,2]] = transformed_points[:,[2,0]]
            

            # Add new columns to the original DataFrame
            # df[["Z_new", "Y_new", "X_new"]] = transformed_points

            # Save the updated CSV
            pd.DataFrame(transformed_points, columns=["axis-0", "axis-1", "axis-2"]).to_csv(newpoints_csv_path,
                                                                                            index=True,
                                                                                            index_label="index")
            # df.to_csv(newpoints_csv_path, index=False)
            print(f"✅ Transformed points saved in {newpoints_csv_path}")

    # SECTION 2

    def process_segmentation_folder(self, input_folder):
        """
        Process all segmentation masks in the input folder and transform the centroids.
        - Detects corresponding reference CSV files.
        - Saves the transformed centroids in a structured folder.

        Parameters:
        - input_folder: Path to the dataset containing images, masks, and reference CSVs.
        """

        # Get all segmentation masks in the folder
        mask_files = [f for f in os.listdir(input_folder) if f.endswith(self.file_identifier + ".tif")]

        for index, mask_file in enumerate(mask_files):
            self.progress_bar.setValue(int(100*index/len(mask_files)))
            base_name = mask_file.replace((self.file_identifier + ".tif"), "")
            mask_path = os.path.join(input_folder, mask_file)
            ref_csv_path = os.path.join(input_folder, base_name + "imaref.csv")

            # Ensure the reference CSV exists
            if not os.path.exists(ref_csv_path):
                print(f"⚠️ Missing reference file for {base_name}, skipping.")
                continue

            # Load segmentation mask
            mask = imread(mask_path)

            # Extract cell centroids using regionprops
            props = regionprops(mask)
            centroids = np.array([prop.centroid for prop in props])

            # Load the reference coordinate system
            origin, u, v, w = load_coordinate_system(ref_csv_path)

            # Transform centroids
            transformed_centroids = transform_points(centroids, origin, u, v, w)
            transformed_centroids = transformed_centroids[:, [2, 1, 0]]

            # Save the transformed centroids
            output_csv_path = os.path.join(input_folder, base_name + "cells_coordinates.csv")
            pd.DataFrame(transformed_centroids, columns=["axis-0", "axis-1", "axis-2"]).to_csv(output_csv_path,
                                                                                            index=True,
                                                                                            index_label="index")

            print(f"✅ Transformed centroids saved: {output_csv_path}")

    # SECTION 3

    def transform_trackmate_res(self, input_folder, format="csv"):
        """
        Process TrackMate CSV file containing spots coordinates across time.

        Parameters:
        - input_folder: Path to the dataset containing point CSVs and reference coordinate CSVs.
        """
        files = os.listdir(input_folder)
        if format == "csv":
            point_files = [f for f in files if f.endswith(self.file_identifier + ".csv")]

            for index, point_file in enumerate(point_files):
                self.progress_bar.setValue(int(100*index/len(point_files)))
                base_name = point_file.replace((self.file_identifier + ".csv"), "")
                tracks_csv_path = os.path.join(input_folder, point_file)
                ref_csv_path = os.path.join(input_folder, base_name + "imaref.csv")

                # Ensure the reference CSV exists
                if not os.path.exists(os.path.join(input_folder, ref_csv_path)):
                    print(f"⚠️ Missing reference file for {base_name}, skipping.")
                    continue

                # Load the reference coordinate system
                origin, u, v, w = load_coordinate_system(ref_csv_path)

                # Load the points CSV
                df = pd.read_csv(tracks_csv_path)

                # Extract point coordinates (Z, Y, X) and reorder them to (X, Y, Z) for consistency
                points = df[["POSITION_Z", "POSITION_Y", "POSITION_X"]].values[3:, :].astype("float32")

                # Transform the points
                transformed_points = transform_points(points, origin, u, v, w)

                header = np.array([["X", "Y", "Z"], ["X", "Y", "Z"], ["(micron)", "(micron)", "(micron)"]])

                # Add new columns to the original DataFrame
                df[["NEW_POSITION_X", "NEW_POSITION_Y", "NEW_POSITION_Z"]] = np.vstack((header, transformed_points))

                # Save the updated CSV
                df.to_csv(tracks_csv_path, index=False)
                print(f"✅ Transformed points saved in {tracks_csv_path}")
        else:
            xml_files = [f for f in files if f.endswith(self.file_identifier + ".xml")]

            for index, xml_file in enumerate(xml_files):
                self.progress_bar.setValue(int(100*index/len(xml_files)))
                base_name = xml_file.replace((self.file_identifier + ".xml"), "")
                xml_path = os.path.join(input_folder, xml_file)
                first_image_index = xml_file.split("<")[1].split("-")[0]
                ref_csv_path = os.path.join(input_folder, xml_file.split("<")[0] + first_image_index + "_imaref.csv")

                # Ensure the reference CSV exists
                if not os.path.exists(ref_csv_path):
                    print(f"⚠️ Missing reference file for {base_name}, skipping.")
                    continue

                # Load the reference coordinate system
                origin, u, v, w = load_coordinate_system(ref_csv_path)

                # Parse the XML file
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Find all spot elements
                spots = root.findall(".//Spot")

                # Extract original coordinates
                original_points = []
                for spot in spots:
                    x = float(spot.attrib["POSITION_X"])
                    y = float(spot.attrib["POSITION_Y"])
                    z = float(spot.attrib["POSITION_Z"])
                    original_points.append([z, y, x])

                # Convert to NumPy array
                original_points = np.array(original_points)

                # Transform the points
                transformed_points = transform_points(original_points, origin, u, v, w)

                # Update XML with new transformed coordinates
                for spot, new_coords in zip(spots, transformed_points):
                    spot.set("NEW_POSITION_Z", str(new_coords[0]))
                    spot.set("NEW_POSITION_Y", str(new_coords[1]))
                    spot.set("NEW_POSITION_X", str(new_coords[2]))

                # Overwrite the original XML file
                tree.write(xml_path, encoding="utf-8", xml_declaration=True)

                print(f"✅ Transformed XML updated in place: {xml_path}")

    def transform_mastodon_csv(self, input_folder):
        """
        Process Mastodon CSV file containing spots coordinates across time.

        Parameters:
        - input_folder: Path to the dataset containing point CSVs and reference coordinate CSVs.
        """
        files = os.listdir(input_folder)

        point_files = [f for f in files if f.endswith(self.file_identifier + ".csv")]

        for index, point_file in enumerate(point_files):
            self.progress_bar.setValue(int(100*index/len(point_files)))
            base_name = point_file.replace((self.file_identifier + ".csv"), "")
            tracks_csv_path = os.path.join(input_folder, point_file)
            ref_csv_path = os.path.join(input_folder, base_name + "_imaref.csv")

            # Ensure the reference CSV exists
            if not os.path.exists(os.path.join(input_folder, ref_csv_path)):
                print(f"⚠️ Missing reference file for {base_name}, skipping.")
                continue

            # Load the reference coordinate system
            origin, u, v, w = load_coordinate_system(ref_csv_path)

            # Load the points CSV
            df = pd.read_csv(tracks_csv_path)

            # Extract point coordinates (Z, Y, X) and reorder them to (X, Y, Z) for consistency
            points = df[["Spot position.2", "Spot position.1", "Spot position"]].values[2:, :].astype("float32")

            # Transform the points
            transformed_points = transform_points(points, origin, u, v, w)

            header = np.array(
                [["NEW_POSITION_X", "NEW_POSITION_Y", "NEW_POSITION_Z"], ["(micron)", "(micron)", "(micron)"]])

            # Add new columns to the original DataFrame
            df[["New Spot position", "New Spot position.1", "New Spot position.2"]] = np.vstack(
                (header, transformed_points))

            # Save the updated CSV
            df.to_csv(tracks_csv_path, index=False)
            print(f"✅ Transformed points saved in {tracks_csv_path}")

    def transform_ultrack_csv(self, input_folder):
        """
        Process Ultrack CSV file containing spots coordinates across time.

        Parameters:
        - input_folder: Path to the dataset containing point CSVs and reference coordinate CSVs.
        """
        files = os.listdir(input_folder)

        point_files = [f for f in files if f.endswith(self.file_identifier + ".csv")]

        for index, point_file in enumerate(point_files):
            self.progress_bar.setValue(int(100*index/len(point_files)))
            base_name = point_file.replace((self.file_identifier + ".csv"), "")
            tracks_csv_path = os.path.join(input_folder, point_file)
            ref_csv_path = os.path.join(input_folder, base_name + "_imaref.csv")

            # Ensure the reference CSV exists
            if not os.path.exists(os.path.join(input_folder, ref_csv_path)):
                print(f"⚠️ Missing reference file for {base_name}, skipping.")
                continue

            # Load the reference coordinate system
            origin, u, v, w = load_coordinate_system(ref_csv_path)

            # Load the points CSV
            df = pd.read_csv(tracks_csv_path)

            # Extract point coordinates (Z, Y, X) and reorder them to (X, Y, Z) for consistency
            points = df[["z", "y", "x"]].values.astype("float32")

            # Transform the points
            transformed_points = transform_points(points, origin, u, v, w)

            # Add new columns to the original DataFrame
            df[["z_new", "y_new", "x_new"]] = transformed_points

            # Save the updated CSV
            df.to_csv(tracks_csv_path, index=False)
            print(f"✅ Transformed points saved in {tracks_csv_path}")

def make_points_transformation_widget(viewer: Viewer) -> CoordinateTransformationWidget:
    return CoordinateTransformationWidget(viewer)
