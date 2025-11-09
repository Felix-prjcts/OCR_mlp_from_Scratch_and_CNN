import numpy as np
import pandas as pd
import matplotlib as plt


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

        self.poids[0] = np.random.randn(nb_neurones[0], 28 * 28 + 1)         *       np.sqrt(1 / (28 * 28))

        for i in range(1, self.nb_couches):
            self.poids[i] = np.random.randn(nb_neurones[i], nb_neurones[i - 1] + 1)       * np.sqrt(1 / nb_neurones[i - 1])

        self.poids[self.nb_couches] = np.random.randn(nb_classes, nb_neurones[-1] + 1)    * np.sqrt(1 / nb_neurones[-1])
        # lignes x colonne
    def ajout_biais(self, pixels):
        return np.append(pixels, 1).reshape(-1, 1)  # Biais = 1

    def image(self, data, nb_lignes):
        labels = data.iloc[:nb_lignes, 0].values
        pixels = data.iloc[:nb_lignes, 1:].values / 255
        return [(label, self.ajout_biais(pixels[i])) for i, label in enumerate(labels)]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def forward(self, entree):
        somme = []
        activations = [self.ajout_biais(entree).reshape(-1, 1)]    #vecteur colonne

        for i in range(self.nb_couches):
            z = np.dot(self.poids[i], activations[-1])    #produit entre matrice des poids et activation précédente
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

    def entrainer(self, dataset):
        for i in range(self.iterations):
            for data in dataset:
                somme, activations = self.forward(data[1])
                self.backward(data[0], data[1], somme, activations)

    def tester(self, pixels):
        _, activations = self.forward(pixels)
        return np.argmax(activations[-1])


nb_classes = 10
nb_neurones = [256, 128, 64]
iterations = 5
taux = 0.03

#mlp = MultiClassPerceptron(nb_classes, nb_neurones, iterations, taux)

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

print(f"Nombre d'images de train : {len(pixel_test)}")
#lp.entrainer(pixel_train)

#print("Entraînement terminé")

#correct_predictions = 0
#total_predictions = 0

#for i in range(1000):
    #testeur = pixel_test[i]
    #prediction = mlp.tester(testeur[1])
    #label = testeur[0]
    #if prediction == label:
     #   correct_predictions += 1
    #total_predictions += 1

#accuracy = correct_predictions / total_predictions * 100
#print(f"Taux de réussite : {accuracy:.2f}%")


learning_rates = [0.001, 0.01, 0.03, 0.5,0.1]

results = {}

for lr in learning_rates :
    mlp = MultiClassPerceptron(nb_classes, nb_neurones, iterations, lr)
    mlp.entrainer(pixel_train)
    correct_predictions = sum(1 for label, pixels in pixel_test if mlp.tester(pixels) == label)
    accuracy = correct_predictions / len(pixel_test) * 100
    results[lr] = accuracy
    print(f"Taux d'apprentissage: {lr}, Précision: {accuracy:.2f}%")

# Tracer le graphe
plt.plot(results.keys(), results.values(), marker='o', linestyle='-')
plt.xscale('log')


architectures = {"2 couches": [256, 128], "3 couches": [256, 128, 64], "4 couches": [256, 128, 64, 32]}
accuracy_results = {}
nb_classes = 10
iterations = 5
taux = 0.03

for nom, arch in architectures.items():
    mlp = MultiClassPerceptron(nb_classes, arch, iterations, taux)
    mlp.entrainer(pixel_train)
    correct_predictions = sum(1 for label, pixels in pixel_test if mlp.tester(pixels) == label)
    accuracy = correct_predictions / len(pixel_test) * 100
    accuracy_results[nom] = accuracy
    print(f"{nom}: {accuracy:.2f}%")

# Tracer les résultats
plt.bar(accuracy_results.keys(), accuracy_results.values(), color=['blue', 'orange', 'green'])
plt.xlabel("Nombre de couches cachées")
plt.ylabel("Précision (%)")
plt.title("Impact du nombre de couches cachées sur la précision")
plt.ylim(0, 100)
plt.show()
