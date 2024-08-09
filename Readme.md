# Napari zebra fish embryo stereotyping plugin

Ce plugin pour Napari permet de sélectionner un dossier contenant des images d'embryon de zebrafish,
de tracer des segments sur chaque image, de calculer l'origine et les vecteurs directeurs de la base orthonormée
définie par ces segments et de sauvegarder les résultats dans des fichiers CSV.

## Fonctionnalités

- Sélectionner un dossier contenant des images.
- Tracer des segments sur les images avec le mode de dessin de segments activé par défaut.
- Sauvegarder les résultats des calculs dans des fichiers CSV portant le même nom que les images.
- Afficher des instructions pour guider l'utilisateur tout au long du processus.

## Prérequis

- Python 3.7+
- Napari
- magicgui
- qtpy
- scikit-image

Vous pouvez installer les dépendances nécessaires avec pip :

```bash
pip install napari[all] magicgui qtpy scikit-image
