import numpy as np

class Nettoyage:
    def __init__(self, matrice, seuil=100):
        self.matrice = matrice
        self.seuil = seuil

    def nettoyer(self):
        image_binaire = (self.matrice <= self.seuil) * 1
        return image_binaire


image_test = np.array([[120, 80, 200, 250, 10],
                       [130, 90, 210, 240, 20],
                       [140, 100, 220, 230, 30],
                       [150, 110, 230, 220, 40],
                       [160, 120, 240, 210, 50]])

# Créer une instance de la classe Nettoyage
nettoyage_image = Nettoyage(image_test, seuil=100)

# Appliquer la méthode de nettoyage
image_binaire = nettoyage_image.nettoyer()

# Afficher le résultat
print(image_binaire)