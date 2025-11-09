import numpy as np
import pandas as pd

mnist_data = pd.read_csv("mnist_train.csv")

mnisttest = pd.read_csv("mnist_test.csv")



class Perceptron:

    def __init__(self, neurones, poids_ini, nombre, parametre):
        self.taux = 0.01
        self.neurones = neurones + [1]
        self.poids = poids_ini + [0.01]
        self.nombre = nombre
        self.parametre = parametre
        self.resultat = None
        self.pond = []

    def image(self, nb_lignes):
        chiffres = mnist_data.iloc[:, [0]].values  # Première colonne : les chiffres
        pixels = mnist_data.iloc[:, 1:].values / 255  # Normaliser les pixels
        tab = []
        for i in range(nb_lignes):
            chiffre = chiffres[i]
            if chiffre == self.parametre:
                resultat = 1
            else:
                resultat = 0
            tab.append((resultat, pixels[i].ravel()))
        return tab

    def neur(self):
        self.pond = [self.neurones[i] * self.poids[i] for i in range(len(self.neurones))]


    def somme(self):
        a = 0
        for i in self.pond:
            a += i
        return self.activ(a)

    def activ(self, a):
        self.resultat = 1 if a > 0 else 0
        return self.resultat

    def correction(self):
        for i in range(len(self.neurones)):
            self.poids[i] += self.taux * (self.nombre - self.resultat) * self.neurones[i]

    def entrainer(self, data, iterations):
        for i in range(iterations):
            for label, pixel in data:
                self.neurones = pixel + [1]
                self.nombre = label
                self.neur()
                self.somme()

                self.correction()

    def tester(self, chiffre):
        self.neurones = chiffre[1]  # Pixel de l'image
        self.nombre = chiffre[0]  # Label attendu
        self.neur()  # Calcul des produits pondérés
        return self.somme()


neurones_initial = np.random.rand(784)
poids_initial = np.random.rand(784) * 0.01
perceptron = Perceptron(neurones_initial, poids_initial, 1, 1)
donnees = perceptron.image(nb_lignes=10000)

perceptron.entrainer(donnees, iterations=5)

exemple_test = donnees[500]
resultat_test = perceptron.tester(exemple_test)

print(f"Résultat prédit : {resultat_test}, Résultat attendu : {exemple_test[0]}")

cpt = 0
reussite = 0


def imagetest(nb_lignes, parametre):
    chiffres = mnisttest.iloc[:, [0]].values  # Première colonne : les chiffres
    pixels = mnisttest.iloc[:, 1:].values / 255  # Normaliser les pixels
    tab = []
    for i in range(nb_lignes):
        chiffre = chiffres[i]
        if chiffre == parametre:
            resultat = 1
        else:
            resultat = 0
        tab.append((resultat, pixels[i].ravel()))
    return tab

train = imagetest(1000,1)

for test in donnees:
    if perceptron.tester(test) == test[0]:
        reussite += 1
    cpt += 1
taux_reussite = (reussite / cpt) * 100
print("taux de reussite",taux_reussite)