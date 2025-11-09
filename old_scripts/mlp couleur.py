import numpy as np
import pandas as pd
import matplotlib as plt
import os
from PIL import Image

chemin_dossier_test = "C:\\Users\\User\\PycharmProjects\\pythonProject6\\.venv\\testing"
chemin_dossier_train = "C:\\Users\\User\\PycharmProjects\\pythonProject6\\.venv\\training"

dataset_gris_train = []

for label in os.listdir(chemin_dossier_train):
    chemin_sous_dossier = os.path.join(chemin_dossier_train, label)

    if os.path.isdir(chemin_sous_dossier):
        for fichier in os.listdir(chemin_sous_dossier):
            chemin_image = os.path.join(chemin_sous_dossier, fichier)
            try:
                with Image.open(chemin_image) as img:
                    pixels = list(img.getdata())  # Liste des pixels de l'image
                    normalized_pixels = [item / 255.0 for sublist in pixels for item in sublist]
                    dataset_gris_train.append((int(label), normalized_pixels))  # Ajouter les pixels normalisés
            except Exception as e:
                print(f"Erreur avec {chemin_image}: {e}")

print(f"{len(dataset_gris_train)} images chargées avec labels.")

dataset_gris_test = []

for label in os.listdir(chemin_dossier_test):
    chemin_sous_dossier = os.path.join(chemin_dossier_test, label)

    if os.path.isdir(chemin_sous_dossier):
        for fichier in os.listdir(chemin_sous_dossier):
            chemin_image = os.path.join(chemin_sous_dossier, fichier)
            try:
                with Image.open(chemin_image) as img:
                    pixels = list(img.getdata())  # Liste des pixels de l'image
                    normalized_pixels = [item / 255.0 for sublist in pixels for item in sublist]
                    dataset_gris_test.append((int(label), normalized_pixels))  # Ajouter les pixels normalisés
            except Exception as e:
                print(f"Erreur avec {chemin_image}: {e}")

print(f"{len(dataset_gris_test)} images chargées avec labels.")
class MultiClassPerceptron:
    def __init__(self, nb_classes, nb_neurones, iterations, taux, activation):
        self.taux = taux
        self.nb_classes = nb_classes
        self.iterations = iterations
        self.nb_couches = len(nb_neurones)
        self.nb_neurones = nb_neurones
        self.activation=activation
        self.poids = {}
        self.taille_entree = 28 * 28 * 3

        self.poids[0] = np.random.randn(nb_neurones[0], self.taille_entree+1)     *       np.sqrt(1 / self.taille_entree)

        for i in range(1, self.nb_couches):
            self.poids[i] = np.random.randn(nb_neurones[i], nb_neurones[i - 1] + 1)       * np.sqrt(1 / nb_neurones[i - 1])

        self.poids[self.nb_couches] = np.random.randn(nb_classes, nb_neurones[-1] + 1)    * np.sqrt(1 / nb_neurones[-1])

    def ajout_biais(self, pixels):
        return np.append(pixels, 1).reshape(-1, 1)  # Biais = 1

    def image(self, data, nb_lignes):
        labels = data.iloc[:nb_lignes, 0].values
        pixels = data.iloc[:nb_lignes, 1:].values / 255
        return [(label, self.ajout_biais(pixels[i])) for i, label in enumerate(labels)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def forward(self, entree):
        somme = []
        activations = [self.ajout_biais(entree).reshape(-1, 1)]

        for i in range(self.nb_couches):
            z = np.dot(self.poids[i], activations[-1])
            somme.append(z)
            if self.activation=="relu":
                a = self.relu(z)
            else :
                a = self.sigmoid(z)
            a = self.ajout_biais(a)

            activations.append(a)

        z_f = np.dot(self.poids[self.nb_couches], activations[-1])
        somme.append(z_f)
        sortie = self.softmax(z_f)
        activations.append(sortie)

        return somme, activations

    def backward(self, label, pixels, somme, activations):
        y_reel = np.zeros((self.nb_classes, 1))
        y_reel[label] = 1

        delta = activations[-1] - y_reel
        deltas = [delta]

        for i in range(self.nb_couches, 0, -1):
            if self.activation=="relu":
                prime = self.relu_prime(somme[i - 1])
            else :
                prime = activations[i][:-1] * (1 - activations[i][:-1])
            delta = np.dot(self.poids[i].T, delta)[:-1] * prime
            deltas.append(delta)

        deltas.reverse()

        for i in range(self.nb_couches + 1):
            self.poids[i] -= self.taux * np.dot(deltas[i], activations[i].T)

    def entrainer(self, data):
        for i in range(self.iterations):
            np.random.shuffle(data)  # mélange les données
            for j in range(0, len(data), 32):
                batch = data[
                        j:j + 32]
                for exemple in batch:
                    somme, activations = self.forward(exemple[1])
                    self.backward(exemple[0], exemple[1], somme, activations)

    def tester(self, pixels):
        _, activations = self.forward(pixels)
        return np.argmax(activations[-1])


nb_classes = 10
nb_neurones = [256, 128, 64]
iterations = 3
taux = 0.05

mlp = MultiClassPerceptron(nb_classes, nb_neurones, iterations, taux,"sigmoid")

pixel_train = dataset_gris_train
pixel_test = dataset_gris_test


print(f"Nombre d'images de train : {len(pixel_train)}")
print(f"Nombre d'images de test : {len(pixel_test)}")

mlp.entrainer(pixel_train)
print(" Entraînement terminé")

correct_predictions = sum(1 for label, pixels in pixel_test if mlp.tester(pixels) == label)
accuracy = correct_predictions / len(pixel_test) * 100
print(f" Taux de réussite : {accuracy:.2f}%")