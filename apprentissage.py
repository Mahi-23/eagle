# Importer OpenCV2 pour le traitement d'image
# Importer os pour le chemin(path) du fichier
import cv2, os

# Importer numpy pour les calculs de matrices
import numpy as np

# Importer Python Image Library (PIL) (bibliothèque d'images)
from PIL import Image

# Création d'histogrammes de motifs binaires locaux pour la reconnaissance des visages
recognizer = cv2.face.LBPHFaceRecognizer_create()

#  Detercter un visage à l'aide du modèle XML
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# Créer une méthode pour obtenir les images et les données d'étiquette
def getImagesAndLabels(path):

    # Obtenir tous les chemins des fichiers
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
    
    # Initialiser un visage vide
    faceSamples=[]
    
    # Initialiser une Id vide
    ids = []

    # Boucle tout le chemin du fichier
    for imagePath in imagePaths:

        # Obtenir l'image et la convertir en niveaux de gris
        PIL_img = Image.open(imagePath).convert('L')

        # PIL image --> au tableau numpy 
        img_numpy = np.array(PIL_img,'uint8')

        # Obtenir l'Id de l'image
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        print(Id)

        # Obtenir le visage à partir des images 
        faces = detector.detectMultiScale(img_numpy)

        # Boucle pour chaque face, ajouter à leur identifiant respectif
        for (x,y,w,h) in faces:

            # Ajouter l'image dans faceSamples
            faceSamples.append(img_numpy[y:y+h,x:x+w])

            # Ajouter l'Id dans Ids
            ids.append(Id)

    # Retourner le tableau de visage et le tableau d'identifiants
    return faceSamples,ids

# Obtenir les visages et les Ids
faces,ids = getImagesAndLabels('BD')

# Former le modèle à l'aide des faces et des identifiants
recognizer.train(faces, np.array(ids))

# Enregistrer le modèle dans apprentissage.yml
recognizer.save('apprentissage/apprentissage.yml')
