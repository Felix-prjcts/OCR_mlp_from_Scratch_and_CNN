import os

import cv2
import numpy as np

from os.path import join

class bdd(): #Utiliser la fonction sortie
    def __init__(self):
        return None
    def verify_archive_structure(self,):
        """Vérifie la structure de l'archive extraite"""
        print("\nVérification de la structure des fichiers...")
        extract_dir = "curated_data"
        found_files = []

        # Recherche récursive de fichiers images
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    found_files.append(join(root, file))

        if not found_files:
            print("Aucune image trouvée. Structure actuelle :")
            os.system(f"tree -L 3 {extract_dir} || ls -R {extract_dir}")
        else:
            print(f"Found {len(found_files)} image files")

        return found_files


    def load_data_alternative(self):
        X, y = [], []
        extract_dir = "curated_data"
        found_files = self.verify_archive_structure()

        for img_path in found_files:
            try:
                # Essai d'extraction du label depuis le chemin
                dir_name = os.path.basename(os.path.dirname(img_path))
                if dir_name.isdigit():
                    char_code = int(dir_name)
                else:
                    # Si les dossiers ne sont pas des codes ASCII
                    char_code = ord(os.path.basename(img_path)[0])  # Premier caractère du nom de fichier

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    X.append(cv2.resize(img, (32, 32)))
                    y.append(char_code)
            except Exception as e:
                print(f"Erreur sur {img_path}: {e}")

        self.X=np.array(X)
        self.Y=np.array(y)

        return self.X, self.Y


    #renvoie les datasets sous formes: x_train,y_train,x_test,y_test
    #Pour utiliser dans le résseau, il est important d'avoir les indices par raport à 0: faire -33 sur les labels
    def sortie(self):
        self.load_data_alternative()
        indices = np.arange(len(self.X))
        np.random.shuffle(indices)
        x_shuffled = self.X[indices]
        y_shuffled = self.Y[indices]

        # Remappe les labels à des indices 0..n_classes-1
        unique_labels = sorted(np.unique(y_shuffled))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_mapped = np.array([label_map[y] for y in y_shuffled])

        split_idx = int(0.8 * len(self.X))
        x_train = x_shuffled[:split_idx]
        y_train = y_mapped[:split_idx]
        x_test = x_shuffled[split_idx:]
        y_test = y_mapped[split_idx:]

        return x_train, y_train, x_test, y_test

    def get_unique_labels(self):
        self.load_data_alternative()
        return sorted(np.unique(self.Y))

def prepare_dataset(X, Y):
    X = X / 255.0
    return [(int(Y[i]), X[i].flatten().reshape(-1, 1)) for i in range(len(X))]  # Flatten avant reshape






class MultiClassPerceptron:
    def __init__(self, nb_classes, nb_neurones, iterations, taux, activation):
        self.taux = taux
        self.nb_classes = nb_classes
        self.iterations = iterations
        self.nb_couches = len(nb_neurones)
        self.nb_neurones = nb_neurones
        self.activation=activation
        self.poids = {}
        self.taille_entree = 32 * 32

        self.poids[0] = np.random.randn(nb_neurones[0], self.taille_entree+1)     *       np.sqrt(1 / self.taille_entree)

        for i in range(1, self.nb_couches):
            self.poids[i] = np.random.randn(nb_neurones[i], nb_neurones[i - 1] + 1)       * np.sqrt(1 / nb_neurones[i - 1])

        self.poids[self.nb_couches] = np.random.randn(nb_classes, nb_neurones[-1] + 1)    * np.sqrt(1 / nb_neurones[-1])

    def ajout_biais(self, pixels):
        return np.vstack([pixels, 1])  # Ajoute 1 à la fin d'un vecteur colonne

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
        if entree.ndim == 1:
            entree = entree.reshape(-1, 1)  # Convertit en colonne si nécessaire
        activations = [self.ajout_biais(entree)]

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
            for j, (label, pixels) in enumerate(data):
                somme, activations = self.forward(pixels)
                self.backward(label, pixels, somme, activations)
            print(f" Itération {i + 1} terminée")

    def tester(self, pixels):
        _, activations = self.forward(pixels)
        return np.argmax(activations[-1])

a = bdd()
x_train, y_train, x_test, y_test = a.sortie()
y_train = y_train - y_train.min()
y_test = y_test - y_test.min()
pixel_train = prepare_dataset(x_train, y_train)
pixel_test = prepare_dataset(x_test, y_test)

labels_uniques = a.get_unique_labels()
nb_classes = len(labels_uniques)
print(nb_classes)
print("test classe",np.max(y_train) + 1)
nb_neurones = [1024, 512, 256]
iterations = 5
taux = 0.01

mlp = MultiClassPerceptron(nb_classes, nb_neurones, iterations, taux,"relu")



print(f"Nombre d'images de train : {len(pixel_train)}")
print(f"Nombre d'images de test : {len(pixel_test)}")

mlp.entrainer(pixel_train)
print(" Entraînement terminé")

correct_predictions = sum(1 for label, pixels in pixel_test if mlp.tester(pixels) == label)
accuracy = correct_predictions / len(pixel_test) * 100
print(f" Taux de réussite : {accuracy:.2f}%")

mot = str(input("Entrez une chaine de caractére (Ex:BONJOUR!)"))

# Créer une correspondance entre les labels remappés et les lettres
# label_map : label d'origine → index remappé
# inverse_map : index remappé → label d'origine
unique_labels = a.get_unique_labels()
inverse_map = {i: unique_labels[i] for i in range(len(unique_labels))}

# Fonction pour retrouver une image pour chaque lettre
def trouver_image_de_lettre(le_char, x_data, y_data):
    code_ascii = ord(le_char)
    # Trouver le label remappé qui correspond à ce code ASCII
    if code_ascii in unique_labels:
        idx_label = unique_labels.index(code_ascii)
        for i in range(len(y_data)):
            if y_data[i] == idx_label:
                return x_data[i].reshape(-1, 1) / 255.0
    return None


print(" Début de la reconnaissance lettre par lettre :")
for lettre in mot:
    pixels = trouver_image_de_lettre(lettre, x_test, y_test)
    if pixels is not None:
        prediction = mlp.tester(pixels)
        lettre_predite = chr(inverse_map[prediction])
        print(
            f"Lettre cible: '{lettre}' → Prédiction MLP: '{lettre_predite}' {'✅' if lettre == lettre_predite else '❌'}")
        plt.imshow(pixels.reshape(32, 32), cmap='gray')
        plt.title(f"Cible: '{lettre}' - Prédit: '{lettre_predite}'")
        plt.axis('off')
        plt.show()
    else:
        print(f"Lettre '{lettre}' non trouvée dans les données de test.")
