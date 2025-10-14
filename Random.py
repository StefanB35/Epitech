import random

# Liste des personnes avec leur niveau
personnes = [
    ("Rayan", 6),
    ("Benoit", 4),
    ("Perrine", 2),
    ("Antoine", 4),
    ("Stéfan", 7),
    ("Mathys", 6.5),
    ("Haniel", 5.5),
    ("Romain", 8.75)
]

# Fonction pour trouver deux groupes de 4 avec des moyennes proches
def split_balanced(personnes):
    best_diff = float('inf')
    best_g1, best_g2 = None, None
    essais = 10000
    for _ in range(essais):
        random.shuffle(personnes)
        g1 = personnes[:4]
        g2 = personnes[4:]
        m1 = sum(x[1] for x in g1) / 4
        m2 = sum(x[1] for x in g2) / 4
        diff = abs(m1 - m2)
        if diff < best_diff:
            best_diff = diff
            best_g1, best_g2 = list(g1), list(g2)
            if best_diff == 0:
                break
    return best_g1, best_g2


# Demander à l'utilisateur le nombre de groupes et de personnes par groupe
while True:
    try:
        nb_groupes = int(input("Combien de groupes ? (par défaut 2) : ") or 2)
        nb_par_groupe = int(input("Combien de personnes par groupe ? (par défaut 4) : ") or 4)
        if nb_groupes * nb_par_groupe == len(personnes):
            break
        else:
            print(f"Erreur : {nb_groupes} groupes de {nb_par_groupe} personnes ne couvrent pas {len(personnes)} personnes.")
    except ValueError:
        print("Entrée invalide. Veuillez entrer un nombre entier.")

# Fonction généralisée pour n groupes équilibrés
def split_balanced_n(personnes, nb_groupes, nb_par_groupe):
    best_diff = float('inf')
    best_split = None
    essais = 10000
    for _ in range(essais):
        random.shuffle(personnes)
        groupes = [personnes[i*nb_par_groupe:(i+1)*nb_par_groupe] for i in range(nb_groupes)]
        moyennes = [sum(x[1] for x in g)/nb_par_groupe for g in groupes]
        diff = max(moyennes) - min(moyennes)
        if diff < best_diff:
            best_diff = diff
            best_split = [list(g) for g in groupes]
            if best_diff == 0:
                break
    return best_split, [sum(x[1] for x in g)/nb_par_groupe for g in best_split]

groupes, moyennes = split_balanced_n(personnes, nb_groupes, nb_par_groupe)
for i, (g, m) in enumerate(zip(groupes, moyennes), 1):
    print(f"Groupe {i} : {g} Moyenne : {round(m,2)}")