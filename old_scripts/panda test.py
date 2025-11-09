import pandas as pds
from math import sqrt
from tkiteasy1 import *

longueur = 1532
largeur = 800
dimMorpion = 800
type = {
    'Normal': {'Normal': 1, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 1, 'Fighting': 2, 'Poison': 1,
               'Ground': 1,
               'Flying': 1, 'Psychic': 1, 'Bug': 1, 'Rock': 0.5, 'Ghost': 0, 'Dragon': 1, 'Dark': 1, 'Steel': 0.5,
               'Fairy': 1},

    'Fire': {'Normal': 1, 'Fire': 0.5, 'Water': 0.5, 'Electric': 1, 'Grass': 2, 'Ice': 2, 'Fighting': 1, 'Poison': 1,
             'Ground': 1,
             'Flying': 1, 'Psychic': 1, 'Bug': 2, 'Rock': 0.5, 'Ghost': 1, 'Dragon': 0.5, 'Dark': 1, 'Steel': 2,
             'Fairy': 1},

    'Water': {'Normal': 1, 'Fire': 2, 'Water': 0.5, 'Electric': 1, 'Grass': 0.5, 'Ice': 1, 'Fighting': 1, 'Poison': 1,
              'Ground': 2,
              'Flying': 1, 'Psychic': 1, 'Bug': 1, 'Rock': 2, 'Ghost': 1, 'Dragon': 0.5, 'Dark': 1, 'Steel': 1,
              'Fairy': 1},

    'Electric': {'Normal': 1, 'Fire': 1, 'Water': 2, 'Electric': 0.5, 'Grass': 0.5, 'Ice': 1, 'Fighting': 1,
                 'Poison': 1, 'Ground': 0,
                 'Flying': 2, 'Psychic': 1, 'Bug': 1, 'Rock': 1, 'Ghost': 1, 'Dragon': 0.5, 'Dark': 1, 'Steel': 1,
                 'Fairy': 1},

    'Grass': {'Normal': 1, 'Fire': 0.5, 'Water': 2, 'Electric': 1, 'Grass': 0.5, 'Ice': 1, 'Fighting': 1, 'Poison': 0.5,
              'Ground': 2,
              'Flying': 0.5, 'Psychic': 1, 'Bug': 0.5, 'Rock': 2, 'Ghost': 1, 'Dragon': 0.5, 'Dark': 1, 'Steel': 0.5,
              'Fairy': 1},

    'Ice': {'Normal': 1, 'Fire': 0.5, 'Water': 0.5, 'Electric': 1, 'Grass': 2, 'Ice': 0.5, 'Fighting': 1, 'Poison': 1,
            'Ground': 2,
            'Flying': 2, 'Psychic': 1, 'Bug': 1, 'Rock': 1, 'Ghost': 1, 'Dragon': 2, 'Dark': 1, 'Steel': 0.5,
            'Fairy': 1},

    'Fighting': {'Normal': 2, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 2, 'Fighting': 1, 'Poison': 0.5,
                 'Ground': 1,
                 'Flying': 0.5, 'Psychic': 0.5, 'Bug': 0.5, 'Rock': 2, 'Ghost': 0, 'Dragon': 1, 'Dark': 2, 'Steel': 2,
                 'Fairy': 0.5},

    'Poison': {'Normal': 1, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 2, 'Ice': 1, 'Fighting': 1, 'Poison': 0.5,
               'Ground': 0.5,
               'Flying': 1, 'Psychic': 1, 'Bug': 1, 'Rock': 0.5, 'Ghost': 0.5, 'Dragon': 1, 'Dark': 1, 'Steel': 0,
               'Fairy': 2},

    'Ground': {'Normal': 1, 'Fire': 2, 'Water': 1, 'Electric': 2, 'Grass': 0.5, 'Ice': 1, 'Fighting': 1, 'Poison': 2,
               'Ground': 1,
               'Flying': 0, 'Psychic': 1, 'Bug': 0.5, 'Rock': 2, 'Ghost': 1, 'Dragon': 1, 'Dark': 1, 'Steel': 2,
               'Fairy': 1},

    'Flying': {'Normal': 1, 'Fire': 1, 'Water': 1, 'Electric': 0.5, 'Grass': 2, 'Ice': 1, 'Fighting': 2, 'Poison': 1,
               'Ground': 1,
               'Flying': 1, 'Psychic': 1, 'Bug': 2, 'Rock': 0.5, 'Ghost': 1, 'Dragon': 1, 'Dark': 1, 'Steel': 0.5,
               'Fairy': 1},

    'Psychic': {'Normal': 1, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 1, 'Fighting': 2, 'Poison': 2,
                'Ground': 1,
                'Flying': 1, 'Psychic': 0.5, 'Bug': 1, 'Rock': 1, 'Ghost': 1, 'Dragon': 1, 'Dark': 0, 'Steel': 0.5,
                'Fairy': 1},

    'Bug': {'Normal': 1, 'Fire': 0.5, 'Water': 1, 'Electric': 1, 'Grass': 2, 'Ice': 1, 'Fighting': 0.5, 'Poison': 0.5,
            'Ground': 1,
            'Flying': 0.5, 'Psychic': 2, 'Bug': 1, 'Rock': 1, 'Ghost': 0.5, 'Dragon': 1, 'Dark': 2, 'Steel': 0.5,
            'Fairy': 0.5},

    'Rock': {'Normal': 1, 'Fire': 2, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 2, 'Fighting': 0.5, 'Poison': 1,
             'Ground': 0.5,
             'Flying': 2, 'Psychic': 1, 'Bug': 2, 'Rock': 1, 'Ghost': 1, 'Dragon': 1, 'Dark': 1, 'Steel': 0.5,
             'Fairy': 1},

    'Ghost': {'Normal': 0, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 1, 'Fighting': 1, 'Poison': 1,
              'Ground': 1,
              'Flying': 1, 'Psychic': 2, 'Bug': 1, 'Rock': 1, 'Ghost': 2, 'Dragon': 1, 'Dark': 0.5, 'Steel': 1,
              'Fairy': 1},

    'Dragon': {'Normal': 1, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 1, 'Fighting': 1, 'Poison': 1,
               'Ground': 1,
               'Flying': 1, 'Psychic': 1, 'Bug': 1, 'Rock': 1, 'Ghost': 1, 'Dragon': 2, 'Dark': 1, 'Steel': 0.5,
               'Fairy': 0},

    'Dark': {'Normal': 1, 'Fire': 1, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 1, 'Fighting': 0.5, 'Poison': 1,
             'Ground': 1,
             'Flying': 1, 'Psychic': 2, 'Bug': 1, 'Rock': 1, 'Ghost': 2, 'Dragon': 1, 'Dark': 0.5, 'Steel': 1,
             'Fairy': 0.5},

    'Steel': {'Normal': 1, 'Fire': 0.5, 'Water': 0.5, 'Electric': 0.5, 'Grass': 1, 'Ice': 2, 'Fighting': 0.5,
              'Poison': 0, 'Ground': 1,
              'Flying': 1, 'Psychic': 1, 'Bug': 1, 'Rock': 2, 'Ghost': 1, 'Dragon': 1, 'Dark': 1, 'Steel': 0.5,
              'Fairy': 2},

    'Fairy': {'Normal': 1, 'Fire': 0.5, 'Water': 1, 'Electric': 1, 'Grass': 1, 'Ice': 1, 'Fighting': 2, 'Poison': 0.5,
              'Ground': 1,
              'Flying': 1, 'Psychic': 1, 'Bug': 1, 'Rock': 1, 'Ghost': 1, 'Dragon': 2, 'Dark': 2, 'Steel': 0.5,
              'Fairy': 1}
}


class Pokemon :

    def __init__(self):
        self.g = ouvrirFenetre(longueur,largeur)
        self.df = pds.read_csv('poke.csv', index_col="Name")
        print(self.df)
        self.pokedex_joueur1 = []
        self.pokedex_joueur2 = []
        self.pokeplacer = {}  #pokemon placer sur le plteau, cle coordonnee, valeur nom du pokemon
        self.nbp = 42
        self.menu()


    def menu(self):
        menu = self.g.afficherImage(0, 0, "./MenuPokemon.png")
        testjeusolo = self.g.dessinerRectangle(950,645,470,55,"white")
        #settings = self.g.afficherImage(1300, 700, "./Settings.png",150,75)
        while True:
            cliquesouris = self.g.attendreClic()
            if 950 < cliquesouris.x < 1420 and 585 < cliquesouris.y < 635:
                self.g.supprimerTout()
                self.Jeu()


            if 950 < cliquesouris.x<1420 and 710<cliquesouris.y<765:
                self.g.supprimerTout()
                self.settings()

    def settings(self):
        settings = menu = self.g.afficherImage(0, 0, "./MenuSettings.png")

        while True:
            cliquesouris = self.g.attendreClic()
            if 14 < cliquesouris.x<75 and 629<cliquesouris.y<689:
                self.g.supprimerTout()
                self.menu()


    def Jeu(self):
        self.init_pokedex()
        self.l1 = [0 for _ in range(9)]
        self.l2 = [0 for _ in range(9)]
        self.l3 = [0 for _ in range(9)]
        self.l4 = [0 for _ in range(9)]
        self.l5 = [0 for _ in range(9)]
        self.l6 = [0 for _ in range(9)]
        self.l7 = [0 for _ in range(9)]
        self.l8 = [0 for _ in range(9)]
        self.l9 = [0 for _ in range(9)]
        # Dictionnaire pour associer les grandes cases aux listes
        self.lists = {
            1: self.l1,
            2: self.l2,
            3: self.l3,
            4: self.l4,
            5: self.l5,
            6: self.l6,
            7: self.l7,
            8: self.l8,
            9: self.l9
        }
        self.ban = {}
        self.win = {}
        self.victoire = {}
        self.MegaGrille = [0 for _ in range(9)]

        for i in range(1,3):
            self.g.placerAuDessous(self.g.dessinerLigne(i * dimMorpion/3, 0, i * dimMorpion/3, dimMorpion , "white",ep=5))
            self.g.placerAuDessous(self.g.dessinerLigne(0,dimMorpion/3 * i, dimMorpion, dimMorpion/3 * i, "white",ep=5))
        for i in range(3):
            for j in range(3):
                self.g.placerAuDessous(self.dessinerPetiteGrille(i, j))
        self.g.dessinerLigne(dimMorpion,605,1532,605,"white",ep=5)
        self.g.placerAuDessous(self.g.dessinerLigne(dimMorpion,0,dimMorpion,dimMorpion,"white",ep=5))
        self.g.placerAuDessous(self.g.dessinerLigne(0,dimMorpion,dimMorpion,dimMorpion,"white",ep=5))
        self.g.placerAuDessous(self.g.dessinerLigne(0,0,0,dimMorpion,"white",ep=5))
        self.g.placerAuDessous(self.g.dessinerLigne(0,0,dimMorpion,0,"white",ep=5))
        encadre = None
        joueur = 1
        cliquable = None
        Text1 = self.g.afficherTexte("Joueur1", dimMorpion / 2 + 800, 10, "green")
        cpt = 0
        self.pokemon = []
        condition = False
        self.afficherpokemon2(cpt)
        carrénoir = self.g.dessinerRectangle(805, 0, 1532 - 805, 600, "black")
        self.afficherpokemon(cpt)
        while True:


            if condition == True:
                if joueur ==1 and len(self.dicoimage1)!=0:
                    self.g.placerAuDessus(carrénoir)
                    for i in self.dicoimage1:
                        self.g.placerAuDessus(i)

                    condition = False
                if joueur ==2 and len(self.dicoimage2) !=0:
                    self.g.placerAuDessus(carrénoir)
                    for i in self.dicoimage2:
                        self.g.placerAuDessus(i)
                    condition = False

            clic = self.g.attendreClic()
            self.win = {}

            if 0 < clic.x < dimMorpion and 0 < clic.y < dimMorpion:


                grande_case, petite_case = self.determinerCase(clic.x, clic.y)
                if grande_case in self.ban:
                    print(f"Vous devez jouer dans une autre grande case. La grande case {grande_case} est bannie.")
                    continue

                if (grande_case == cliquable and encadre!=None and len(self.pokemon)!=0):
                    if self.lists[grande_case][petite_case-1]==0:

                        self.g.supprimer(encadre)
                        encadre = None


                if cliquable is not None and grande_case != cliquable :
                    print(f"Vous devez jouer dans la grande case {cliquable}.")

                    continue

                if len(self.pokemon)!=0 and self.RemplirGrille(grande_case, petite_case, joueur) :

                    self.Regle(grande_case, joueur)
                    cliquable = self.Transfert(grande_case,petite_case)
                    joueur = 3 - joueur
                    condition = True
                    self.pokemon = []
                    if joueur == 2:
                        self.g.supprimer(self.dicoimage1[self.i])
                        del self.dicopoke1[self.co]
                    if joueur == 1:
                        self.g.supprimer(self.dicoimage2[self.i])
                        del self.dicopoke2[self.co]

                    self.g.supprimer(Text1)
                    cpt+=1
                    if joueur == 1:
                        Text1 = self.g.afficherTexte("Joueur1", dimMorpion / 2 + 800, 10, "green")
                    else:
                        Text1 = self.g.afficherTexte("Joueur2", dimMorpion / 2 + 800, 10, "green")

                    if cliquable is not None :
                        xc,yc = self.dessinerEncadrement(cliquable)
                        encadre = self.g.dessinerRectangle(xc,yc,dimMorpion/3,dimMorpion/3,col="purple")
                        self.g.placerAuDessous(encadre)

            else:
                self.choixpoke(joueur,clic)


    def Regle(self,grande_case,joueur):
        x = None
        if joueur ==1 :
            x =1
        else :
            joueur = 2
        liste = self.lists[grande_case]
        if ((liste[0] % 2 == liste[1] % 2 == liste[2] % 2 and liste[0] != 0 and liste[1] != 0 and liste[2] != 0) or (liste[3] % 2 == liste[4] % 2 == liste[5] % 2 and liste[3] != 0 and liste[4] != 0 and liste[5] != 0) or (liste[6] % 2 == liste[7] % 2 == liste[8] % 2 and liste[6] != 0 and liste[7] != 0 and liste[8] != 0) or (liste[0] % 2 == liste[4] % 2 == liste[8] % 2 and liste[0] != 0 and liste[4] != 0 and liste[8] != 0) or (liste[0] % 2 == liste[3] % 2 == liste[6] % 2 and liste[0] != 0 and liste[3] != 0 and liste[6] != 0) or (liste[1] % 2 == liste[4] % 2 == liste[7] % 2 and liste[1] != 0 and liste[4] != 0 and liste[7] != 0) or (liste[2] % 2 == liste[5] % 2 == liste[8] % 2 and liste[2] != 0 and liste[5] != 0 and liste[8] != 0) or (liste[6] % 2 == liste[4] % 2 == liste[2] % 2 and liste[6] != 0 and liste[4] != 0 and liste[2] != 0)):
            self.MegaGrille[grande_case-1] = joueur
            self.ban[grande_case] = self.lists[grande_case]
            self.win[grande_case] = self.lists[grande_case]
            self.victoire[grande_case] = joueur
            print(self.ban)
            self.dessinerCroix(grande_case, joueur)
            self.Finale()


    def Finale(self):
        if ((self.MegaGrille[0] == self.MegaGrille[1] == self.MegaGrille[2]) and self.MegaGrille[0]!=0 ) or ((self.MegaGrille[3] == self.MegaGrille[4] == self.MegaGrille[5]) and self.MegaGrille[3]!=0 ) or ((self.MegaGrille[6] == self.MegaGrille[7] == self.MegaGrille[8]) and self.MegaGrille[6]!=0 ) or ((self.MegaGrille[0] == self.MegaGrille[4] == self.MegaGrille[8]) and self.MegaGrille[0]!=0 ) or ((self.MegaGrille[0] == self.MegaGrille[3] == self.MegaGrille[6]) and self.MegaGrille[0]!=0 ) or ((self.MegaGrille[1] == self.MegaGrille[4] == self.MegaGrille[7]) and self.MegaGrille[1]!=0) or ((self.MegaGrille[2] == self.MegaGrille[5] == self.MegaGrille[8]) and self.MegaGrille[2]!=0) or ((self.MegaGrille[6] == self.MegaGrille[4] == self.MegaGrille[2]) and self.MegaGrille[6]!=0) :
            self.g.afficherTexte("FINITOOOO", dimMorpion/2, dimMorpion/2,"purple")
            self.g.attendreClic()
            self.g.supprimerTout()

            self.menu()


    def dessinerCroix(self, grande_case, joueur):
        taille = dimMorpion / 3
        colonne = (grande_case -1) % 3
        ligne = (grande_case -1)  // 3
        x = (taille * colonne )
        y = (taille* ligne )
        x1 = x+taille
        y1 = y+taille
        yc= (y1+y)/2
        xc = (x1 + x) / 2
        couleur = "royalblue" if joueur == 1 else "tomato"
        forme = "x" if joueur ==1 else "o"
        self.g.dessinerRectangle(x+3,y+3,taille-5,taille-5, couleur)
        self.g.afficherTexte(forme,xc,yc-(taille/8),"white",sizefont = int(taille))


    def dessinerEncadrement(self, grande_case):
        TaillGC = dimMorpion / 3
        colonne_grande = (grande_case - 1) % 3
        ligne_grande = (grande_case - 1) // 3
        x0 = colonne_grande * TaillGC
        y0 = ligne_grande * TaillGC
        return x0,y0


    def Affichage(self,grande_case,petite_case, joueur):
        taille_grande_case = dimMorpion / 3
        taille_petite_case = taille_grande_case / 3
        # Indices de la grande case
        ligne_grande = (grande_case - 1) // 3
        colonne_grande = (grande_case - 1) % 3
        # Indices de la petite case
        ligne_petite = (petite_case - 1) // 3
        colonne_petite = (petite_case - 1) % 3
        # Calcul des coordonnées du centre
        x = (colonne_grande * taille_grande_case + colonne_petite * taille_petite_case
             + taille_petite_case / 2)
        y = (ligne_grande * taille_grande_case + ligne_petite * taille_petite_case
             + taille_petite_case / 2)
        xc = x - taille_petite_case/2
        yc = y - taille_petite_case/2
        if joueur == 1 :
            if self.lists[grande_case][petite_case-1] == 1:
                self.g.afficherImage(xc+10,yc+5, f"./pokefront/{self.select[0]}.png",70,70)
                self.g.placerAuDessous(self.g.dessinerRectangle(xc,yc,taille_petite_case,taille_petite_case,"royalblue"))
            elif self.lists[grande_case][petite_case-1] == 3:
                self.g.dessinerRectangle(xc+1, yc+1, taille_petite_case-1, taille_petite_case-1, "royalblue")
                self.g.afficherTexte("X", x, y, "white", sizefont=int(taille_petite_case / 2))
        if joueur == 2 :
            if self.lists[grande_case][petite_case - 1] == 2:
                self.g.afficherImage(xc+10,yc+5, f"./pokefront/{self.select[0]}.png",70,70)
                self.g.placerAuDessous(self.g.dessinerRectangle(xc,yc,taille_petite_case,taille_petite_case,"tomato"))
            elif self.lists[grande_case][petite_case-1]==4:
                self.g.dessinerRectangle(xc+1, yc+1, taille_petite_case-1, taille_petite_case-1, "tomato")
                self.g.afficherTexte("O", x, y, "white", sizefont=int(taille_petite_case / 2))

    def Transfert(self,grande_case, petite_case):
        liste = self.lists[petite_case]
        if petite_case in self.ban:
            return None
        if grande_case in self.win:
            return petite_case
        if grande_case in self.ban:
            return None
        if all(i !=0 for i in liste) :
            return None
        else :
            return petite_case

    def RemplirGrille(self, grande_case, petite_case, joueur):
        print("cest au tour du joueur",joueur)
        print(self.pokeplacer)
        J = None
        adv = None
        if joueur % 2 == 0:
            J = 2
            adv = 1
        else:
            J = 1
            adv = 2 # Verification
        if self.lists[grande_case][petite_case - 1] == (3 or 4):
            print(f"Case {petite_case} dans la grande case {grande_case} déjà remportée !")
            return False

        if self.lists[grande_case][petite_case - 1] == 0:
            self.lists[grande_case][petite_case - 1] = joueur
            self.placerpokemon(grande_case,petite_case,joueur)
            print(self.pokeplacer)
            self.Affichage(grande_case,petite_case, joueur)
            return True

        if self.lists[grande_case][petite_case - 1] == joueur:
            print("Vous occuper deja cette case")
            return False

        if self.lists[grande_case][petite_case - 1] == adv:
            attaquant = self.pokemon[0]
            if self.combat(attaquant, grande_case, petite_case, joueur) == True:
                self.lists[grande_case][petite_case-1] = J+2
                gagnant = J
            else :
                self.lists[grande_case][petite_case-1] = adv+2
                print(self.lists[grande_case][petite_case-1])
                gagnant = adv
            print("le gagnant est le pokemon du joueur",gagnant)
            self.Affichage(grande_case,petite_case,gagnant)
            return True

    def init_pokedex(self):
        self.dicopoke1 = {}
        self.dicopoke2 = {}
        joueur1_df = self.df.sample(n=self.nbp,random_state=1)  # Pokémon du joueur 1
        joueur2_df = self.df.sample(n=self.nbp,random_state=2)  # Pokémon du joueur 2

        self.numpokedexj1 = joueur1_df["#"].tolist()
        self.numpokedexj2 = joueur2_df["#"].tolist()

        # Stocker les noms des Pokémon
        self.pokedex_joueur1 = joueur1_df.index.tolist()
        self.pokedex_joueur2 = joueur2_df.index.tolist()

        self.dicopoke1[1] = self.numpokedexj1
        self.dicopoke2[1]= self.numpokedexj2
        print(self.dicopoke1[1],self.dicopoke1)
        print("Pokédex du Joueur 1 :")
        print(self.pokedex_joueur1)

        print("Pokédex du Joueur 2 :")
        print(self.pokedex_joueur2)

    def choixpoke(self, joueur,clic):
        self.pokemon = []
        self.select = []
        if joueur == 1:
            print(self.pokedex_joueur1)
            poke = self.pokedex_joueur1
            dico = self.dicopoke1
            num = self.numpokedexj1
        else:
            print(self.pokedex_joueur2)
            poke = self.pokedex_joueur2
            dico = self.dicopoke2
            num = self.numpokedexj2

        nb_pokemon = self.nbp
        entier = int(sqrt(nb_pokemon))
        if entier < sqrt(nb_pokemon):
            entier += 1
        largeur_cellule = 732 / entier  # Largeur des cellules
        hauteur_cellule = 600 / entier  # Hauteur des cellules
        for cle in dico:
            if cle != 1:
                x, y = cle

                x_min = x - largeur_cellule / 2
                x_max = x + largeur_cellule / 2
                y_min = y - hauteur_cellule / 2
                y_max = y + hauteur_cellule / 2

                if x_min < clic.x < x_max and y_min < clic.y < y_max:
                    choix=dico[cle]
                    self.i = poke.index(choix)
                    self.select.append(num[self.i])
                    self.pokemon.append(choix)
                    self.co = (x,y)
                    print(f"Pokémon sélectionné : {choix}")


    def afficherpokemon(self,indice):
        self.dicoimage1 = []

        espace_droit = 732  # Largeur disponible à droite de la grille
        hauteur_totale = 600
        nb_pokemon = self.nbp

        entier = int(sqrt(nb_pokemon))
        if entier < sqrt(nb_pokemon):
            entier += 1

        largeur_cellule = espace_droit / entier
        hauteur_cellule = hauteur_totale / entier

        taille_image = int(min(largeur_cellule, hauteur_cellule) * 0.8)

        cpt = 0
        for l in range(entier):
            for c in range(entier):
                if cpt < nb_pokemon and cpt - indice < len(self.pokedex_joueur1):
                    x = 800 + (c+0.1)* largeur_cellule
                    y = (l + 0.5) * hauteur_cellule
                    print("X ET Y DES POKE ",(x,y))
                    self.dicopoke1[(x+largeur_cellule/2, y+hauteur_cellule/2)] = self.pokedex_joueur1[cpt-indice]
                    self.dicoimage1.append(self.g.afficherImage(x, y,f"./pokefront/{self.dicopoke1[1][cpt]}.png",taille_image, taille_image))

                    cpt += 1

        return True



    def afficherpokemon2(self, indice):
        self.dicoimage2 = []
        espace_droit = 732  # Largeur disponible à droite de la grille
        hauteur_totale = 600
        nb_pokemon = self.nbp

        entier = int(sqrt(nb_pokemon))
        if entier < sqrt(nb_pokemon):
            entier += 1

        largeur_cellule = espace_droit / entier
        hauteur_cellule = hauteur_totale / entier

        taille_image = int(min(largeur_cellule, hauteur_cellule) * 0.8)

        cpt = 0
        for l in range(entier):
            for c in range(entier):
                if cpt < nb_pokemon and cpt - indice < len(self.pokedex_joueur2):
                    x = 800 + (c + 0.1) * largeur_cellule
                    y = (l + 0.5) * hauteur_cellule

                    self.dicopoke2[(x + largeur_cellule / 2, y + hauteur_cellule / 2)] = self.pokedex_joueur2[cpt-indice]
                    self.dicoimage2.append(self.g.afficherImage(x, y,
                                         f"./pokefront/{self.dicopoke2[1][cpt]}.png",
                                         taille_image, taille_image))

                    cpt += 1

        return True

    def placerpokemon(self,grande_case,petite_case,joueur):  #Associe pokemon a une case
        #pokemon = self.choixpoke(joueur)
        pokemon = self.pokemon[0]
        self.pokeplacer[(grande_case,petite_case)] = pokemon
        print("la case", (grande_case,petite_case), "est occupé par le pokemon", pokemon)

    def multiplicateur(self,pokeatt,pokedef):
        multiplicateur=1
        typeatt = pokeatt['Type 1']
        typedef1 = pokedef['Type 1']
        typedef2 = pokedef['Type 2']
        multiplicateur *= type[typeatt][typedef1]
        if pokedef['Type 2'] in type:
            multiplicateur*=type[typeatt][typedef2]
        return multiplicateur


    def combat(self, pokeatt, grande_case, petite_case, joueur):
    # Récupérer le nom du Pokémon défenseur
        pokedef = self.pokeplacer.get((grande_case, petite_case))
        if pokedef is None:
            print(f"Aucun Pokémon trouvé dans la case {grande_case}, {petite_case}.")
            return False

        # Vérifier si les noms existent dans le DataFrame


        # Extraire les statistiques des Pokémon depuis le DataFrame
        pokeattstat = self.df.loc[pokeatt]
        pokedefstat = self.df.loc[pokedef]

        pokeattHP = pokeattstat['HP']
        pokedefHP = pokedefstat['HP']

        multiatt = self.multiplicateur(pokeattstat,pokedefstat)
        multidef = self.multiplicateur(pokedefstat,pokeattstat)
        print(f"Multiplicateur Charizard => Mawile {multiatt}")
        print(f"Multiplicateur Mawile =>  Charizard {multidef}")
        pokeattatt = max(pokeattstat['Attack'],pokeattstat['Sp. Atk'])
        pokedefatt = max(pokedefstat['Attack'],pokedefstat['Sp. Atk'])

        if pokedefatt == pokedefstat['Attack']:
            pokeattdef = pokeattstat['Defense']
        else:
            pokeattdef = pokeattstat['Sp. Def']

        if pokeattatt == pokeattstat['Attack']:
            pokedefdef = pokedefstat['Defense']
        else:
            pokedefdef = pokedefstat['Sp. Def']

        print(pokedefatt)
        print(pokedefdef)
        print(pokeattatt)
        print(pokeattdef)


        while pokeattHP > 0 and pokedefHP > 0:
        # Attaque de l'attaquant
            degats = (((((50*0.4)+2)*pokeattatt*80)/pokedefdef)/50 +2)*multiatt
            pokedefHP -= degats
            print(f"{pokeatt} attaque {pokedef} infligeant {degats} dégâts. HP restant de {pokedef} : {max(0, pokedefHP)}")



        # Attaque du défenseur
            degats = (((((50*0.4)+2)*pokedefatt*80)/pokeattdef)/50 +2)*multidef
            pokeattHP -= degats
            print(f"{pokedef} attaque {pokeatt} infligeant {degats} dégâts. HP restant de {pokeatt} : {max(0, pokeattHP)}")

        if pokedefHP <=0 and pokeattHP <=0:
            return False

        if pokedefHP <= 0:
            print(f"{pokedef} est K.O.!")
            return True

        if pokeattHP <= 0:
            print(f"{pokeatt} est K.O.!")
            return False  # Défenseur gagne

    def dessinerPetiteGrille(self, i, j):
        x0 = j * dimMorpion / 3
        y0 = i * dimMorpion / 3
        x1 = x0 + dimMorpion / 3
        y1 = y0 + dimMorpion / 3

        for k in range(1, 3):
            self.g.dessinerLigne(x0 + k * (dimMorpion / 9), y0, x0 + k * (dimMorpion / 9), y1, "white")
            self.g.dessinerLigne(x0, y0 + k * (dimMorpion / 9), x1, y0 + k * (dimMorpion / 9), "white")



    def determinerCase(self, x, y):
        taille_case_grande_x = dimMorpion / 3
        taille_case_grande_y = dimMorpion / 3
        taille_case_petite_x = dimMorpion / 9
        taille_case_petite_y = dimMorpion / 9

        # Trouver la grande case
        colonne_grande = int(x // taille_case_grande_x)
        ligne_grande = int(y // taille_case_grande_y)
        numero_grande_case = ligne_grande * 3 + colonne_grande + 1

        # Trouver la petite case dans la grande case
        x_relative = x % taille_case_grande_x
        y_relative = y % taille_case_grande_y
        colonne_petite = int(x_relative // taille_case_petite_x)
        ligne_petite = int(y_relative // taille_case_petite_y)
        numero_petite_case = ligne_petite * 3 + colonne_petite + 1

        return numero_grande_case, numero_petite_case





Pokemon()