#!/usr/bin/env python3
"""
Clean_couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-france copy.py
Nettoyage des données de couverture vaccinale France depuis 2011

DESCRIPTION:
    Script de nettoyage pour les données nationales françaises de couverture 
    vaccinale des adolescents et adultes depuis 2011. Version de traitement
    spécifique pour les données agrégées au niveau national.

FONCTIONNALITÉS:
    - Suppression de colonnes spécifiques HPV et méningocoque
    - Sauvegarde automatique du fichier original
    - Traitement des données temporelles (2011 à aujourd'hui)
    - Préservation de l'encodage UTF-8
    - Sortie standardisée vers Data_Clean

SPÉCIFICITÉ:
    Traite les données au niveau national français (agrégation de toutes les régions)
    contrairement aux versions départementales ou régionales.

USAGE:
    python "Clean_couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-france copy.py"

AUTEUR: Stéfan Beaulieu
DATE: 2025
"""

# =============================================================================
# IMPORTS ET CONFIGURATION
# =============================================================================
from pathlib import Path
import shutil
import pandas as pd
import sys


# =============================================================================
# CONFIGURATION DES DONNÉES NATIONALES FRANÇAISES
# =============================================================================

# Fichier source - données nationales France depuis 2011
INPUT = Path("Hackathon/Data/couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-france.csv")

# Validation de l'existence du fichier
if not INPUT.exists():
    print(f"Fichier source introuvable: {INPUT}")
    sys.exit(1)

# Colonnes à exclure de l'analyse (identiques à la version départementale)
# Focus sur les vaccinations HPV et méningocoque spécifiques
TO_DROP = [
    "HPV filles 1 dose à 15 ans",      # Papillomavirus - filles 1ère dose
    "HPV filles 2 doses à 16 ans",     # Papillomavirus - filles rappel
    "HPV garçons 1 dose à 15 ans",     # Papillomavirus - garçons 1ère dose
    "HPV garçons 2 doses à 16 ans",    # Papillomavirus - garçons rappel
    "Méningocoque C 10-14 ans",        # Méningocoque C - 10-14 ans
    "Méningocoque C 15-19 ans",        # Méningocoque C - 15-19 ans
    "Méningocoque C 20-24 ans",        # Méningocoque C - 20-24 ans
]

print("🇫🇷 Traitement des données nationales françaises de couverture vaccinale")
print(f"Période couverte: depuis 2011")
print(f"Fichier source: {INPUT}")

# Sauvegarde préventive du fichier original
bak = INPUT.with_suffix(INPUT.suffix + ".bak")
shutil.copy2(INPUT, bak)
print(f"Sauvegarde créée: {bak}")

# Chargement, suppression et écriture
df = pd.read_csv(INPUT, encoding="utf-8")
present = [c for c in TO_DROP if c in df.columns]

# Définir le chemin de sortie dans Data_Clean
output_dir = Path("Hackathon/Data_Clean")
output_dir.mkdir(parents=True, exist_ok=True)  # créer le dossier si nécessaire
output_file = output_dir / "couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-region-clean.csv"

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