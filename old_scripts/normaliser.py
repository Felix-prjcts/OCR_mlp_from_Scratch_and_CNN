import numpy as np

class Normaliser: #Normalise les image obtenu apres le découpage en liste applati de 784 pixels
    def __init__(self, images_lettres, dim=(28, 28)): #prend en entree un ensemble de matrices carees
        # Liste pour stocker les images normalisées
        self.images_normal = []

        # Traiter chaque image dans la liste
        for image_lettre in images_lettres:
            image_normal = self.carre(image_lettre)  # Applique la transformation pour rendre l'image carrée
            image_normal = self.resize(image_normal, dim)  # Applique ensuite le redimensionnement sur l'image carrée
            image_normal = image_normal / 255.0  # Normaliser l'image
            image_normal = image_normal.reshape(-1)  # Aplatir l'image en une liste de 784 pixels
            self.images_normal.append(image_normal)

    def resize(self, image_lettre, dim): #redim l'image en 28,28 et renvoie une matrice
        h, l = image_lettre.shape
        nouv_h, nouv_l = dim

        nouvelle_image = np.zeros((nouv_h, nouv_l), dtype=np.float32)

        h_ratio = h / nouv_h
        l_ratio = l / nouv_l

        for i in range(nouv_h):
            for j in range(nouv_l):
                # Trouver la position correspondante sur l'image d'origine
                x = int(j * h_ratio)
                y = int(i * l_ratio)
                nouvelle_image[i, j] = image_lettre[x, y]

        return nouvelle_image #renvoie la liste de 784 pixels

    def carre(self,image_lettre):  # Redimensionner l'image pour la rendre carrée en ajoutant des bordures blanches
        h, l = image_lettre.shape
        size = max(h, l)  # La nouvelle taille carrée sera égale à la plus grande dimension

        image_carre = np.ones((size, size), dtype=np.uint8) * 255

        x = (size - l) // 2 #coin gauche x
        y = (size - h) // 2 #coin gauche y
        image_carre[y:y + h, x:x + l] = image_lettre

        return image_carre


    def renvoyer_image(self): #renvoie une liste de tout les images sous forme de liste applatit (784 pixels)
        return np.array(self.images_normal)


