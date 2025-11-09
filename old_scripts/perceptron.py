import numpy as np
import pandas as pd

# Chargement des données
mnist_data = pd.read_csv("mnist_train.csv")
mnisttest = pd.read_csv("mnist_test.csv")

#Truc a faire varier test, fct activation, iterations, taux apprentissage, normalisation
class MultiClassPerceptron:
    def __init__(self, nb_classes, nb_neur,lr ):
        self.taux = lr
        self.nb_classes = nb_classes
        self.poids = np.random.rand(nb_classes, nb_neur + 1) *0.01  # Inclut le biais

    def ajout_biais(self, pixels):
        return np.append(pixels, 1)  # Ajout du biais comme dernier élément

    def image(self, data, nb_lignes):
        labels = data.iloc[:nb_lignes, 0].values  # Première colonne : labels
        pixels = data.iloc[:nb_lignes, 1:].values / 255  # Normaliser les pixels
        return [(label, self.ajout_biais(pixels[i])) for i, label in enumerate(labels)]

    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps)

    def somme(self, neurones):
        return self.softmax(np.dot(self.poids, neurones))  # Sigmoid appliqué aux produits scalaires

    def activ(self, somme):
        return np.argmax(somme)  # Classe avec la probabilité maximale

    def correction(self, neurones, label):
        scores = self.somme(neurones)
        for classe in range(self.nb_classes):
            cible = 1 if classe == label else 0  # Cible pour la classe
            erreur = cible - scores[classe]
            self.poids[classe] += self.taux * erreur * neurones

    def entrainer(self, data, iterations):
        for i in range(iterations):
            for label, pixels in data:
                self.correction(pixels, label)

    def tester(self, pixels):
        return self.activ(self.somme(pixels))


# Initialisation
nb_classes = 10
nb_neur = 784
perceptron = MultiClassPerceptron(nb_classes, nb_neur, lr=0.001)

# Entraînement
donnees_train = perceptron.image(mnist_data, nb_lignes=10000)
perceptron.entrainer(donnees_train, iterations=40) #A partir de plus de 50 itérations: réussite de 91%

# Test
donnees_test = perceptron.image(mnisttest, nb_lignes=10000)
reussite = sum(1 for label, pixels in donnees_test if perceptron.tester(pixels) == label)
taux_reussite = (reussite / len(donnees_test)) * 100
print(f"Taux de réussite : {taux_reussite:.2f}%")

#Bruit Gaussien

def bruit_gaussien(images,std, mean=0 ):
    bruit = np.random.normal(mean, std, images.shape)
    images_bruit = []

    for i in range(len(images)):
        combi = images[i] + bruit[i]
        # Contraindre les valeurs entre 0 et 1
        combi = np.where(combi > 1, 1, combi)  # Si >1, remplacer par 1
        combi = np.where(combi < 0, 0, combi)  # Si <0, remplacer par 0
        images_bruit.append(combi)

    return np.array(images_bruit)

pixels_test = mnisttest.iloc[:, 1:].values / 255
pixels_bruit5 = bruit_gaussien(pixels_test,0.05,0)
pixels_b_biais5 = [np.append(pixels, 1) for pixels in pixels_bruit5]
labels = mnisttest.iloc[:10000, 0].values
tab_bruit5 = [(label, pixels_b_biais5[i]) for i, label in enumerate(labels)]

reussite_bruit = sum(
    1 for label, pixels in tab_bruit5 if perceptron.tester(pixels) == label
)
taux_reussite_bruit1 = (reussite_bruit / len(tab_bruit5)) * 100
print(f"Taux de réussite avec bruit gaussien d'écart type de 5% : {taux_reussite_bruit1}, soit un ecart de {taux_reussite-taux_reussite_bruit1}%")

#Bruit gaussien de 10%
pixels_bruit10 = bruit_gaussien(pixels_test,0.10,0)
pixels_b_biais10 = [np.append(pixels, 1) for pixels in pixels_bruit10]
tab_bruit10 = [(label, pixels_b_biais10[i]) for i, label in enumerate(labels)]

reussite_bruit = sum(
    1 for label, pixels in tab_bruit10 if perceptron.tester(pixels) == label
)
taux_reussite_bruit10 = (reussite_bruit / len(tab_bruit10)) * 100
print(f"Taux de réussite avec bruit gaussien d'écart type de 10% : {taux_reussite_bruit10}, soit un ecart de {taux_reussite-taux_reussite_bruit10}%")

#Bruit gaussien de 15%
pixels_bruit15 = bruit_gaussien(pixels_test,0.15,0)
pixels_b_biais15 = [np.append(pixels, 1) for pixels in pixels_bruit15]
tab_bruit15 = [(label, pixels_b_biais15[i]) for i, label in enumerate(labels)]

reussite_bruit = sum( 1 for label, pixels in tab_bruit15 if perceptron.tester(pixels) == label)
taux_reussite_bruit15 = (reussite_bruit / len(tab_bruit15)) * 100
print(f"Taux de réussite avec bruit gaussien d'écart type de 15% : {taux_reussite_bruit15}, soit un ecart de {taux_reussite-taux_reussite_bruit15}%")


def bruit_saturation(images, nb_pixels, valeur=1, positions=None):
    if positions is None:
        indices = np.random.choice(images.size, nb_pixels, replace=False)
        print(indices)
    else:
        indices = positions
    np.put(images, indices, valeur)  # Saturer les pixels aux indices choisis
    return images

pixels_test = mnisttest.iloc[:, 1:].values / 255
labels = mnisttest.iloc[:10000, 0].values

#pixels_sature = [(label, bruit_saturation(pixel.copy(), len(pixel))) for label, pixel in zip(labels, pixels_test)]

#print("Pixels saturés:", pixels_sature[:5])  # Afficher les premiers éléments pour vérifier

def saturation(images, nb_pixels, valeur=0, positions=None):
    imageb = images.copy()
    for img in imageb:
        if positions is None:
            indices = np.random.choice(img.size, nb_pixels, replace=False)
        else:
            indices = positions[:nb_pixels]
        np.put(img, indices, valeur)
    return imageb

pixels_test = mnisttest.iloc[:, 1:].values / 255

pixels_bruit_aleatoire = saturation(pixels_test, nb_pixels=100)

pixels_b_biais_aleatoire = [perceptron.ajout_biais(pixels) for pixels in pixels_bruit_aleatoire]
tab_bruit_aleatoire = [(label, pixels_b_biais_aleatoire[i]) for i, label in enumerate(labels)]
reussite_bruit_aleatoire = sum(
    1 for label, pixels in tab_bruit_aleatoire if perceptron.tester(pixels) == label
)
taux_reussite_bruit_aleatoire = (reussite_bruit_aleatoire / len(tab_bruit_aleatoire)) * 100

print(f"Taux de réussite avec bruit aléatoire sur 100 pixels : {taux_reussite_bruit_aleatoire:.2f}%")

pixels_bruit_aleatoire_50 = saturation(pixels_test, nb_pixels=50)
pixels_b_biais_aleatoire_50 = [perceptron.ajout_biais(pixels) for pixels in pixels_bruit_aleatoire_50]
tab_bruit_aleatoire_50 = [(label, pixels_b_biais_aleatoire_50[i]) for i, label in enumerate(labels)]
reussite_bruit_aleatoire_50 = sum(
    1 for label, pixels in tab_bruit_aleatoire_50 if perceptron.tester(pixels) == label
)
taux_reussite_bruit_aleatoire_50 = (reussite_bruit_aleatoire_50 / len(tab_bruit_aleatoire_50)) * 100
print(f"Taux de réussite avec bruit aléatoire sur 50 pixels : {taux_reussite_bruit_aleatoire_50:.2f}%")

# Saturation avec 200 pixels
pixels_bruit_aleatoire_200 = saturation(pixels_test, nb_pixels=200)
pixels_b_biais_aleatoire_200 = [perceptron.ajout_biais(pixels) for pixels in pixels_bruit_aleatoire_200]
tab_bruit_aleatoire_200 = [(label, pixels_b_biais_aleatoire_200[i]) for i, label in enumerate(labels)]
reussite_bruit_aleatoire_200 = sum(
    1 for label, pixels in tab_bruit_aleatoire_200 if perceptron.tester(pixels) == label
)
taux_reussite_bruit_aleatoire_200 = (reussite_bruit_aleatoire_200 / len(tab_bruit_aleatoire_200)) * 100
print(f"Taux de réussite avec bruit aléatoire sur 200 pixels : {taux_reussite_bruit_aleatoire_200:.2f}%")