#!/usr/bin/env python3
"""
Clean_couvertures-vaccinales-des-adolescent-et-adultes-departement.py
Nettoyage des données de couverture vaccinale par département

DESCRIPTION:
    Script de nettoyage spécialisé pour les données de couverture vaccinale 
    des adolescents et adultes par département. Supprime des colonnes spécifiques
    liées aux vaccinations HPV et méningocoque pour simplifier l'analyse.

FONCTIONNALITÉS:
    - Suppression de colonnes spécifiques de vaccination
    - Sauvegarde automatique du fichier original (.bak)
    - Création automatique du dossier de sortie Data_Clean
    - Gestion des cas où certaines colonnes n'existent pas
    - Encodage UTF-8 préservé

COLONNES SUPPRIMÉES:
    - HPV filles/garçons (1 dose à 15 ans, 2 doses à 16 ans)
    - Méningocoque C (tranches d'âge 10-14, 15-19, 20-24 ans)

USAGE:
    python Clean_couvertures-vaccinales-des-adolescent-et-adultes-departement.py

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
# CONFIGURATION DES FICHIERS ET COLONNES
# =============================================================================

# Fichier d'entrée - données de couverture vaccinale par département
INPUT = Path("Hackathon/Data/couvertures-vaccinales-des-adolescent-et-adultes-departement.csv")

# Vérification de l'existence du fichier d'entrée
if not INPUT.exists():
    print(f"Fichier introuvable: {INPUT}")
    sys.exit(1)

# Liste des colonnes à supprimer pour simplifier l'analyse
# Ces colonnes concernent des vaccinations spécifiques qui ne sont pas 
# nécessaires pour l'analyse principale
TO_DROP = [
    "HPV filles 1 dose à 15 ans",      # Vaccination HPV filles - 1ère dose
    "HPV filles 2 doses à 16 ans",     # Vaccination HPV filles - rappel
    "HPV garçons 1 dose à 15 ans",     # Vaccination HPV garçons - 1ère dose  
    "HPV garçons 2 doses à 16 ans",    # Vaccination HPV garçons - rappel
    "Méningocoque C 10-14 ans",        # Méningocoque C - adolescents jeunes
    "Méningocoque C 15-19 ans",        # Méningocoque C - adolescents âgés
    "Méningocoque C 20-24 ans",        # Méningocoque C - jeunes adultes
]

# =============================================================================
# PROCESSUS DE NETTOYAGE
# =============================================================================

print("Début du processus de nettoyage des données de couverture vaccinale")
print(f"Fichier source: {INPUT}")

# Sauvegarde du fichier original avec extension .bak
bak = INPUT.with_suffix(INPUT.suffix + ".bak")
shutil.copy2(INPUT, bak)
print(f"Sauvegarde créée: {bak}")

# Chargement du fichier CSV avec encodage UTF-8
print("Chargement des données...")
df = pd.read_csv(INPUT, encoding="utf-8")
print(f"Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")

# Vérification de la présence des colonnes à supprimer
present = [c for c in TO_DROP if c in df.columns]
print(f"Colonnes trouvées à supprimer: {len(present)}/{len(TO_DROP)}")

# Définition du chemin de sortie dans le dossier Data_Clean
output_dir = Path("Hackathon/Data_Clean")
output_dir.mkdir(parents=True, exist_ok=True)  # Créer le dossier si nécessaire
output_file = output_dir / "couvertures-vaccinales-des-adolescent-et-adultes-departement-clean.csv"

# Traitement conditionnel selon la présence des colonnes
if present:
    # Suppression des colonnes identifiées
    df = df.drop(columns=present)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Colonnes supprimées: {present}")
    print(f"Fichier nettoyé sauvegardé dans: {output_file}")
    print(f"Résultat: {len(df)} lignes, {len(df.columns)} colonnes")
    print(f"Original sauvegardé en {bak}")
else:
    # Aucune colonne à supprimer, mais copie vers Data_Clean quand même
    df.to_csv(output_file, index=False, encoding="utf-8")
    print("Aucune des colonnes demandées n'a été trouvée.")
    print(f"Fichier copié sans modification dans: {output_file}")

print("Processus de nettoyage terminé avec succès")