from pathlib import Path
import shutil
import pandas as pd
import sys

#!/usr/bin/env python3
# GitHub Copilot
# Nettoie le CSV en supprimant les colonnes demandées et sauvegarde dans Data_Clean


INPUT = Path("Hackathon/Data/couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-france.csv")
if not INPUT.exists():
    print(f"Fichier introuvable: {INPUT}")
    sys.exit(1)

# Colonnes à supprimer
TO_DROP = [
    "HPV filles 1 dose à 15 ans",
    "HPV filles 2 doses à 16 ans",
    "HPV garçons 1 dose à 15 ans",
    "HPV garçons 2 doses à 16 ans",
    "Méningocoque C 10-14 ans",
    "Méningocoque C 15-19 ans",
    "Méningocoque C 20-24 ans",
]

# Sauvegarde du fichier original
bak = INPUT.with_suffix(INPUT.suffix + ".bak")
shutil.copy2(INPUT, bak)

# Chargement, suppression et écriture
df = pd.read_csv(INPUT, encoding="utf-8")
present = [c for c in TO_DROP if c in df.columns]

# Définir le chemin de sortie dans Data_Clean
output_dir = Path("Hackathon/Data_Clean")
output_dir.mkdir(parents=True, exist_ok=True)  # créer le dossier si nécessaire
output_file = output_dir / "couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-france-clean.csv"

if present:
    df = df.drop(columns=present)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Colonnes supprimées: {present}")
    print(f"Fichier nettoyé sauvegardé dans: {output_file}")
    print(f"Original sauvegardé en {bak}")
else:
    # Même si aucune colonne n'est supprimée, sauvegarder dans Data_Clean
    df.to_csv(output_file, index=False, encoding="utf-8")
    print("Aucune des colonnes demandées n'a été trouvée.")
    print(f"Fichier copié sans modification dans: {output_file}")