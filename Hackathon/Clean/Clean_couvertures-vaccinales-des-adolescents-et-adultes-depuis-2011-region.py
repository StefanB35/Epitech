#!/usr/bin/env python3
"""
Clean_couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-region.py
Nettoyage des données de couverture vaccinale régionale depuis 2011

DESCRIPTION:
    Script de nettoyage pour les données régionales de couverture vaccinale
    des adolescents et adultes depuis 2011. Traite les données par région
    française pour permettre des analyses géographiques détaillées.

FONCTIONNALITÉS:
    - Suppression de colonnes spécifiques HPV et méningocoque
    - Conservation des données par région française
    - Sauvegarde automatique et sécurisée
    - Préparation pour analyses géospatiales
    - Encodage UTF-8 maintenu

NIVEAU GÉOGRAPHIQUE:
    Régions françaises (niveau intermédiaire entre national et départemental)
    
PÉRIODE:
    Données historiques depuis 2011 jusqu'à présent

USAGE:
    python Clean_couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-region.py

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
# CONFIGURATION DES DONNÉES RÉGIONALES
# =============================================================================

# Fichier source - données régionales depuis 2011
# Note: Le chemin semble pointer vers le fichier france (possiblement une erreur à corriger)
INPUT = Path("Hackathon/Data/couvertures-vaccinales-des-adolescents-et-adultes-depuis-2011-france.csv")

# Validation de l'existence du fichier source
if not INPUT.exists():
    print(f"Fichier source non trouvé: {INPUT}")
    sys.exit(1)

# Colonnes de vaccination spécifique à exclure pour simplifier l'analyse
TO_DROP = [
    "HPV filles 1 dose à 15 ans",      # Vaccination HPV filles première dose
    "HPV filles 2 doses à 16 ans",     # Vaccination HPV filles deuxième dose
    "HPV garçons 1 dose à 15 ans",     # Vaccination HPV garçons première dose
    "HPV garçons 2 doses à 16 ans",    # Vaccination HPV garçons deuxième dose
    "Méningocoque C 10-14 ans",        # Méningocoque C tranche 10-14 ans
    "Méningocoque C 15-19 ans",        # Méningocoque C tranche 15-19 ans
    "Méningocoque C 20-24 ans",        # Méningocoque C tranche 20-24 ans
]

print("Traitement des données régionales de couverture vaccinale")
print(f"Données historiques depuis 2011")
print(f"Fichier source: {INPUT}")

# Création de la sauvegarde de sécurité
bak = INPUT.with_suffix(INPUT.suffix + ".bak")
shutil.copy2(INPUT, bak)
print(f"Sauvegarde de sécurité: {bak}")

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