import numpy as np
import pandas as pd

mnist_data = pd.read_csv("mnist_train.csv")  # Données d'entraînement
mnisttest = pd.read_csv("mnist_test.csv")  # Données de test

class MultiClassPerceptron:
    def __init__(self, nb_classes, nb_neurones, iterations, taux):
        self.taux = taux
        self.nb_classes = nb_classes
        self.iterations = iterations
        self.nb_couches = len(nb_neurones)
        self.nb_neurones = nb_neurones

        self.poids = {}

        self.poids[0] = np.random.randn(nb_neurones[0], 28 * 28 + 1)     *       np.sqrt(1 / (28 * 28))

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

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def forward(self, entree):
        somme = []
        activations = [self.ajout_biais(entree).reshape(-1, 1)]

        for i in range(self.nb_couches):
            z = np.dot(self.poids[i], activations[-1])
            somme.append(z)
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
            sigmoide_prime = activations[i][:-1] * (1 - activations[i][:-1])
            delta = np.dot(self.poids[i].T, delta)[:-1] * sigmoide_prime
            deltas.append(delta)

        deltas.reverse()

        for i in range(self.nb_couches + 1):
            self.poids[i] -= self.taux * np.dot(deltas[i], activations[i].T)

    def entrainer(self, data):
        for i in range(self.iterations):
            np.random.shuffle(data)  
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
iterations = 10
taux = 0.05

mlp = MultiClassPerceptron(nb_classes, nb_neurones, iterations, taux)


pixel_train = []

for i in range(5000):
    pixels = mnisttest.iloc[i, 1:].values / 255
    label = mnisttest.iloc[i, 0]
    pixel_train.append((label, pixels))

pixel_test = []

for i in range(5000, len(mnisttest)):
    pixels = mnisttest.iloc[i, 1:].values / 255
    label = mnisttest.iloc[i, 0]
    pixel_test.append((label, pixels))

print(f"Nombre d'images de test : {len(pixel_test)}")

mlp.entrainer(pixel_train)

print("Entraînement terminé")

correct_predictions = 0
total_predictions = 0

for i in range(1000):
    testeur = pixel_test[i]
    prediction = mlp.tester(testeur[1])
    label = testeur[0]
    if prediction == label:
        correct_predictions += 1
    total_predictions += 1

accuracy = correct_predictions / total_predictions * 100
print(f"Taux de réussite : {accuracy:.2f}%")