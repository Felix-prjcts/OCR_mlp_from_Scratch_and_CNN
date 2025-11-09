import pandas as pd
import numpy as np

# Chargement des données
mnist_data = pd.read_csv("mnist_train.csv") #data d'entrainement
mnisttest = pd.read_csv("mnist_test.csv") #data de test


#####################################################################
                      #PERCEPTRON CLASSIQUE#
#####################################################################

#Perceptron permettant de reconnaitre un chiffre choisi

class Perceptron():

    def __init__(self,chiffre_sol,iterations,sigmoid,multiplicateur_poids,proba,taux):
        self.chiffre_sol=chiffre_sol
        self.taux = taux
        self.poids_solo = np.random.rand(784) * multiplicateur_poids +[0.01]
        self.iterations=iterations
        self.sig=sigmoid
        self.proba = proba

    def image_solo(self, data, nb_lignes):
        labels = data.iloc[:nb_lignes, 0].values  # Première colonne : labels
        pixels = data.iloc[:nb_lignes, 1:].values / 255  # Normaliser les pixels
        tab=[]
        for i in range(nb_lignes):
            chiffre = labels[i]
            if chiffre == self.chiffre_sol:
                resultat = 1
            else:
                resultat = 0
            tab.append((resultat, pixels[i].ravel()))
        return tab

    def neur(self,neurones):
        pond = [neurones[i] * self.poids_solo[i] for i in range(len(neurones))]
        return pond

    def somme_solo(self,pond):
        a = 0
        for i in pond:
            a += i
        return a

    def activ_solo(self, a):
        resultat = 1 if a > 0 else 0
        return resultat

    def correction_solo(self,neurones,nombre,resultat):
        for i in range(len(neurones)):
            self.poids_solo[i] += self.taux * (nombre - resultat) * neurones[i]


    def somme_sigmoid(self, pond):
        return np.sum(pond)  # Somme pondérée des neurones

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))  # Fonction sigmoïde

    def activ_sigmoid(self, a):
        if self.sigmoid(a)> self.proba:
            return 1
        else :
            return  0



    def train_solo(self,data):
        for i in range(self.iterations):
            for label, pixel in data:
                pixel=pixel+[1]
                pond=self.neur(pixel)
                if self.sig==False:
                    a=self.somme_solo(pond)
                    resultat=self.activ_solo(a)
                    self.correction_solo(pixel,label,resultat)
                else:
                    a = self.somme_sigmoid(pond)
                    resultat = self.activ_sigmoid(a)  # Appliquer la sigmoïde
                    self.correction_solo(pixel, label, resultat)


        def tester_solo(self,chiffre):
        neurones = chiffre[1]  # Pixel de l'image
        pond=self.neur(neurones)# Calcul des produits pondérés
        a=self.somme_solo(pond)
        return self.activ_solo(a)



#####################################################################
                    #PERCEPTRON multiclasses#
#####################################################################

#Perceptron capable de détecter n'importe quel chiffre

class MultiClassPerceptron:
    def __init__(self, nb_classes, nb_neur,iterations,multiplicateur_poids,taux):
        self.taux = taux
        self.nb_classes = nb_classes
        self.iterations=iterations
        self.poids = np.random.rand(nb_classes, nb_neur + 1) * multiplicateur_poids  # Inclut le biais

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
        #produit matriciel

    def activ(self, somme):
        return np.argmax(somme)  # Classe avec la probabilité maximale

    def correction(self, neurones, label):
        scores = self.somme(neurones)
        for classe in range(self.nb_classes):
            cible = 1 if classe == label else 0  # Cible pour la classe
            erreur = cible - scores[classe]
            self.poids[classe] += self.taux * erreur * neurones

    def entrainer(self, data):
        for i in range(self.iterations):
            for label, pixels in data:
                self.correction(pixels, label)

    def tester(self, pixels):
        return self.activ(self.somme(pixels))


#####################################################################
                #INITIALISATION DES PERCEPTRONS#
#####################################################################

#Initialisation perceptron classique
chiffre_choisi=7
perceptron_solo=Perceptron(chiffre_choisi,iterations=2,sigmoid=False,multiplicateur_poids=0.01,proba=0.95,taux=0.001)
#Simple avec sigmoid
perceptron_sigmoide = Perceptron(chiffre_choisi,iterations=2,sigmoid=True,multiplicateur_poids=0.01,proba=0.95,taux=0.001)

# Initialisation perceptron amélioré
nb_classes = 10
nb_neur = 784
perceptron = MultiClassPerceptron(nb_classes, nb_neur,iterations=2,multiplicateur_poids=0.01,taux=0.001)



#####################################################################
                #ENTRAINEMENT DES PERCEPTRONS#
#####################################################################

#Entrainement perceptron classic
donnees_train_solo=perceptron_solo.image_solo(mnist_data, nb_lignes=10000)
perceptron_solo.train_solo(donnees_train_solo)

#Entrainement perceptron sigmoid
donnees_train_sigmo = perceptron_sigmoide.image_solo(mnist_data, nb_lignes=10000)
perceptron_sigmoide.train_solo(donnees_train_sigmo)

# Entraînement perceptron amélioré
donnees_train = perceptron.image(mnist_data, nb_lignes=10000)
perceptron.entrainer(donnees_train) #A partir de plus de 50 itérations: réussite de 91%

#####################################################################
                   #TESTS DES PERCEPTRONS#
####################################################################

#Tests des perceptrons classiques
donnees_test_solo=perceptron_solo.image_solo(mnisttest,nb_lignes=10000)
cpt = 0
reussite_classique = 0
reussite_sigmoide = 0
for test in donnees_test_solo:
    resultat_classique = perceptron_solo.tester_solo(test)
    resultat_sigmoide = perceptron_sigmoide.tester_solo(test)

    if resultat_classique == test[0]:
        reussite_classique += 1
    if resultat_sigmoide == test[0]:
        reussite_sigmoide += 1
    cpt += 1
taux_reussite_solo = (reussite_classique / cpt) * 100
taux_reussite_sigmo = (reussite_sigmoide / cpt) * 100
print(f"Taux de réussite avec fonction seuil : {taux_reussite_solo:.2f}%")
print(f"Taux de réussite avec fonction sigmoid : {taux_reussite_sigmo:.2f}%")

# Test Perceptron amélioré
donnees_test = perceptron.image(mnisttest, nb_lignes=10000)
reussite = sum(1 for label, pixels in donnees_test if perceptron.tester(pixels) == label)
taux_reussite = (reussite / len(donnees_test)) * 100
print(f"Taux de réussite : {taux_reussite:.2f}%")


#####################################################################
                   #BRUIT GAUSSIEN#
#####################################################################


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


taux_reussite_bruit1 = (reussite_bruit / len(tab_bruit5)) * 100
print(f"Taux de réussite avec bruit gaussien d'écart type de 5% : {taux_reussite_bruit1_solo}, soit un ecart de {taux_reussite-taux_reussite_bruit1_solo}%")



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

def saturation(images, nb_pixels, valeur=1, positions=None):
    imageb = images.copy()
    for img in imageb:
        if positions is None:
            indices = np.random.choice(img.size, nb_pixels, replace=False)
        else:
            indices = positions[:nb_pixels]
        np.put(img, indices, 1)
    return imageb

pixels_test = mnisttest.iloc[:, 1:].values / 255

pixels_bruit_aleatoire = saturation(pixels_test, nb_pixels=100)
print(pixels_test[:1])
print(pixels_bruit_aleatoire[:1])
pixels_b_biais_aleatoire = [perceptron.ajout_biais(pixels) for pixels in pixels_bruit_aleatoire]
tab_bruit_aleatoire = [(label, pixels_b_biais_aleatoire[i]) for i, label in enumerate(labels)]
reussite_bruit_aleatoire = sum(
    1 for label, pixels in tab_bruit_aleatoire if perceptron.tester(pixels) == label
)
taux_reussite_bruit_aleatoire = (reussite_bruit_aleatoire / len(tab_bruit_aleatoire)) * 100

print(f"Taux de réussite avec bruit aléatoire sur 100 pixels : {taux_reussite_bruit_aleatoire:.2f}%")