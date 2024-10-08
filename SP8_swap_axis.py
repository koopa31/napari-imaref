import os
import numpy as np
from skimage.io import imread, imsave
import argparse

"""
Le script inberse l'axe des canaux et du z pour les images au SP8.
Il se lance de la manière suivante :
python SP8_swap_axis.py -p /chemin/dossier/image
"""

parser = argparse.ArgumentParser(description='Segmentation of cells images')
parser.add_argument('-p', '--path', metavar='N', type=str, help='Path to the folder containing the images to process')

args = parser.parse_args()

input_path = args.path

# Création du dossier de résultats
res_dir = os.path.join(input_path, "swaped_images")
if os.path.isdir(res_dir) is False:
    os.mkdir(res_dir)

# Liste de toutes les acquisitions
acquisitions = [os.path.join(input_path, f) for f in os.listdir(input_path) if
                os.path.isfile(os.path.join(input_path, f)) and os.path.join(input_path, f).endswith(".tif")]

for p in acquisitions:
    im = imread(p)
    im = np.transpose(im, (1, 0, 2, 3))
    imsave(os.path.join(res_dir, os.path.split(p)[1]), im)

print("Images swappées")
