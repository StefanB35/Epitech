#!/usr/bin/env python3
"""
Clean_finalPredActes.py - Nettoyage des données de prédictions d'actes médicaux

DESCRIPTION:
    Script de nettoyage et normalisation des données de prédictions d'actes médicaux.
    Ce script traite les données pour les préparer à l'analyse et à la modélisation.

FONCTIONNALITÉS:
    - Normalisation des noms de colonnes
    - Détection automatique de l'encodage et des séparateurs
    - Extraction des informations de région et groupe depuis les colonnes catégorielles  
    - Conversion et validation des types de données
    - Suppression des valeurs négatives dans les colonnes numériques
    - Nettoyage des chaînes de caractères
    - Suppression des lignes et colonnes largement vides
    - Élimination des doublons

USAGE:
    python Clean_finalPredActes.py --input "chemin/vers/finalPredActes.csv" 
                                  --output "chemin/vers/finalPredActes_cleaned.csv"

EXEMPLE:
    python Clean_finalPredActes.py --input "Hackathon/Data/finalPredActes.csv" 
                                  --output "Hackathon/Data_Clean/finalPredActes_cleaned.csv"

AUTEUR: Stéfan Beaulieu
DATE: 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================
from pathlib import Path
import argparse
import logging
import re
import unicodedata
import pandas as pd
import numpy as np

# Configuration du logging pour suivre le processus de nettoyage
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# =============================================================================
# FONCTIONS UTILITAIRES DE NORMALISATION
# =============================================================================

def normalize_colname(s: str) -> str:
    """
    Normalise les noms de colonnes pour assurer la cohérence.
    
    Cette fonction transforme les noms de colonnes en format standardisé:
    - Conversion en minuscules
    - Remplacement des espaces et tirets par des underscores
    - Suppression de la ponctuation
    - Normalisation Unicode pour gérer les accents
    
    Args:
        s (str): Nom de colonne original
        
    Returns:
        str: Nom de colonne normalisé
        
    Example:
        normalize_colname("Taux d'Urgences (%)") -> "taux_d_urgences"
    """
    # Conversion en chaîne et suppression des espaces en début/fin
    s = str(s).strip()
    
    # Normalisation Unicode (décomposition des caractères accentués)
    s = unicodedata.normalize("NFKD", s)
    
    # Suppression de la ponctuation (garde seulement lettres, chiffres, espaces, tirets)
    s = re.sub(r"[^\w\s-]", "", s)
    
    # Conversion en minuscules
    s = s.lower()
    
    # Remplacement des espaces et tirets multiples par un seul underscore
    s = re.sub(r"[\s-]+", "_", s)
    
    # Suppression des underscores multiples consécutifs
    s = re.sub(r"_+", "_", s)
    
    # Suppression des underscores en début et fin
    return s.strip("_")


def clean_string_value(v):
    """
    Nettoie et normalise une valeur de type chaîne de caractères.
    
    Cette fonction effectue un nettoyage complet des chaînes:
    - Gestion des valeurs manquantes (NaN)
    - Suppression des espaces en début/fin
    - Normalisation Unicode
    - Suppression des caractères de contrôle
    - Réduction des espaces multiples
    
    Args:
        v: Valeur à nettoyer (peut être string, numeric, ou NaN)
        
    Returns:
        str ou NaN: Valeur nettoyée ou NaN si l'entrée était NaN
        
    Note:
        - Préserve les valeurs NaN pour maintenir l'intégrité des données manquantes
        - Convertit automatiquement les types non-string en string avant nettoyage
    """
    # Préserver les valeurs manquantes
    if pd.isna(v):
        return v
    
    # Conversion en string si nécessaire
    if not isinstance(v, str):
        v = str(v)
    
    # Suppression des espaces en début et fin
    v = v.strip()
    
    # Normalisation Unicode pour la cohérence des caractères
    v = unicodedata.normalize("NFKC", v)
    
    # Suppression des caractères de contrôle (non-printables)
    v = re.sub(r"[\x00-\x1f\x7f]", "", v)
    
    # Réduction des espaces multiples à un seul espace
    v = re.sub(r"\s+", " ", v)
    return v


def detect_date_columns(cols):
    """
    Détecte automatiquement les colonnes qui contiennent des dates.
    
    Cette fonction identifie les colonnes potentiellement temporelles en 
    cherchant des mots-clés spécifiques dans les noms de colonnes.
    
    Args:
        cols: Liste des noms de colonnes à analyser
        
    Returns:
        list: Liste des noms de colonnes identifiées comme contenant des dates
        
    Note:
        - Supporte les mots-clés en français et anglais
        - Inclut les variations courantes (jour/day, mois/month, etc.)
        - Case-insensitive grâce à la normalisation préalable des noms
    """
    # Mots-clés pour identifier les colonnes de dates (français + anglais)
    date_keywords = [
        "date", "jour", "day", "mois", "month", "semaine", "week", 
        "année", "year", "année_predite", "annee", "annee_predite"
    ]
    
    # Recherche des colonnes contenant ces mots-clés
    return [col for col in cols if any(keyword in col for keyword in date_keywords)]


# =============================================================================
# FONCTION PRINCIPALE DE NETTOYAGE
# =============================================================================

def main(input_path: Path, output_path: Path):
    """
    Fonction principale orchestrant le processus complet de nettoyage des données.
    
    Cette fonction coordonne toutes les étapes du nettoyage:
    1. Lecture du fichier avec détection automatique d'encodage
    2. Normalisation des noms de colonnes
    3. Nettoyage des données
    4. Conversion des types
    5. Sauvegarde du résultat
    
    Args:
        input_path (Path): Chemin vers le fichier CSV d'entrée
        output_path (Path): Chemin vers le fichier CSV de sortie nettoyé
    """
    # ==========================================================================
    # ÉTAPE 1: VALIDATION ET LECTURE DU FICHIER D'ENTRÉE
    # ==========================================================================
    
    # Vérification de l'existence du fichier d'entrée
    if not input_path.exists():
        logging.error("Le fichier d'entrée n'existe pas: %s", input_path)
        return

    logging.info("Début du nettoyage de: %s", input_path)
    
    # Lecture avec détection automatique de l'encodage et des séparateurs
    try:
        # Tentative principale: UTF-8 avec détection automatique du séparateur
        df = pd.read_csv(input_path, sep=None, engine="python", encoding="utf-8")
        logging.info("Fichier lu avec encodage UTF-8")
    except Exception:
        try:
            # Tentative de fallback: latin1 (compatible Windows)
            df = pd.read_csv(input_path, engine="python", encoding="latin1")
            logging.info("Fichier lu avec encodage latin1 (fallback)")
        except Exception as e:
            logging.error("Impossible de lire le fichier: %s", e)
            return
    
    logging.info("Données initiales: %d lignes, %d colonnes", len(df), len(df.columns))
    
    # ==========================================================================
    # ÉTAPE 2: NORMALISATION DES NOMS DE COLONNES
    # ==========================================================================
    
    # Sauvegarde des noms originaux et création des noms normalisés
    original_cols = list(df.columns)
    new_cols = [normalize_colname(c) for c in original_cols]
    
    # Suppression du préfixe "region_" des noms de colonnes pour simplifier
    new_cols = [col.replace("region_", "") if col.startswith("region_") else col for col in new_cols]
    
    # Application des nouveaux noms de colonnes normalisés
    rename_map = dict(zip(original_cols, new_cols))
    df.rename(columns=rename_map, inplace=True)
    logging.info("Colonnes normalisées et préfixe 'region_' supprimé: %s", new_cols)

    # drop empty columns
    n_rows = len(df)
    col_missing_frac = df.isna().sum() / max(1, n_rows)
    drop_cols = col_missing_frac[col_missing_frac > 0.9].index.tolist()
    if drop_cols:
        logging.info("Dropping columns with >90%% missing: %s", drop_cols)
        df.drop(columns=drop_cols, inplace=True)

    # trim and clean string columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].map(clean_string_value)

    # Replace "1" values with column name in first 14 columns
    # These columns should be treated as categorical/text columns
    first_14_cols = df.columns[:14]
    for col in first_14_cols:
        # Convert column to string type to accept text values
        df[col] = df[col].astype(str)
        
        # Replace values that are exactly "1", "1.0" or similar with the column name
        mask = (df[col].isin(['1', '1.0', '1.00']))
        if mask.any():
            df.loc[mask, col] = col
            logging.info("Replaced %d values of '1'/'1.0' with column name '%s' in text column", mask.sum(), col)
        
        # Ensure column remains as object/string type
        df[col] = df[col].astype('object')

    # Create a 'region' column based on the first 14 columns
    # Find which column contains the region name (not "0" or "0.0")
    region_cols = df.columns[:14]  # First 14 columns are regions/groups
    
    def get_region_name(row):
        for col in region_cols:
            value = str(row[col])
            # If the value is not "0", "0.0", "nan", etc., it's likely the region name
            if value not in ['0', '0.0', '0.00', 'nan', 'None', '']:
                return value
        return 'unknown'  # fallback if no region found
    
    df['region'] = df.apply(get_region_name, axis=1)
    logging.info("Created 'region' column based on first 14 columns")
    
    # Create a 'groupe' column based on the group columns in the first 14 columns
    # Find columns that contain group information (with "groupe_" in name)
    group_cols = [col for col in region_cols if 'groupe_' in col]
    
    def get_groupe_name(row):
        for col in group_cols:
            if col in row.index:
                value = str(row[col])
                # If the value is not "0", "0.0", "nan", etc., it's likely the group name
                if value not in ['0', '0.0', '0.00', 'nan', 'None', '']:
                    return value
        return 'unknown'  # fallback if no group found
    
    df['groupe'] = df.apply(get_groupe_name, axis=1)
    logging.info("Created 'groupe' column based on group columns: %s", group_cols)
    
    # Remove the first 14 columns since we now have the region and group info in separate columns
    cols_to_drop = df.columns[:14].tolist()
    df = df.drop(columns=cols_to_drop)
    logging.info("Dropped first 14 columns: %s", cols_to_drop)

    # Force specific column types
    # Regional and age group columns should be strings
    region_and_age_cols = [
        'ile_de_france', 'centre_val_de_loire', 'bourgogne_et_franche_comte', 
        'normandie', 'hauts_de_france', 'grand_est', 'pays_de_la_loire', 
        'bretagne', 'nouvelle_aquitaine', 'occitanie', 'auvergne_et_rhone_alpes', 
        'corse', 'groupe_moins_de_65_ans', 'groupe_65_ans_et_plus'
    ]
    
    for col in region_and_age_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            logging.info("Forced column '%s' to string type", col)
    
    # region and groupe columns should also be strings
    if 'region' in df.columns:
        df['region'] = df['region'].astype(str)
        logging.info("Forced column 'region' to string type")
    
    if 'groupe' in df.columns:
        df['groupe'] = df['groupe'].astype(str)
        logging.info("Forced column 'groupe' to string type")
    
    # annee, annee_predite and code columns should be integers
    int_cols = ['annee', 'annee_predite', 'code', 'actes', 'doses']
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            logging.info("Forced column '%s' to integer type", col)

    # detect and parse date columns
    date_cols = detect_date_columns(df.columns)
    for c in date_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            logging.info("Parsed date column: %s", c)
        except Exception:
            logging.debug("Could not parse date column: %s", c)

    # coerce numeric-like columns (skip the ones we already forced)
    skip_cols = region_and_age_cols + int_cols + date_cols + ['region', 'groupe']
    for c in df.columns:
        if c in skip_cols:
            continue
        # if majority of values look numeric, coerce
        sample = df[c].dropna().astype(str).head(100)
        num_like = sample.str.replace(r"[^\d\.\-\+,eE]", "", regex=True).str.match(r"^[\+\-]?\d*\.?\d+(e[\+\-]?\d+)?$").sum()
        if len(sample) > 0 and num_like / len(sample) > 0.6:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")

    # Replace negative values with 0 in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            df.loc[df[col] < 0, col] = 0
            logging.info("Replaced %d negative values with 0 in column '%s'", negative_count, col)

    # drop rows that are almost entirely empty
    row_missing_frac = df.isna().sum(axis=1) / max(1, len(df.columns))
    drop_rows_idx = row_missing_frac[row_missing_frac > 0.5].index
    if len(drop_rows_idx) > 0:
        logging.info("Dropping %d rows with >50%% missing", len(drop_rows_idx))
        df.drop(index=drop_rows_idx, inplace=True)

    # drop exact duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    logging.info("Dropped %d duplicate rows", before - len(df))

    # reset index
    df.reset_index(drop=True, inplace=True)

    # write cleaned file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logging.info("Wrote cleaned CSV to %s (rows: %d, cols: %d)", output_path, len(df), len(df.columns))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Clean finalPredActes.csv")
    p.add_argument("--input", "-i", type=Path, default=Path("Hackathon/Data/finalPredActes.csv"))
    p.add_argument("--output", "-o", type=Path, default=Path("Hackathon/Data_Clean/finalPredActes_cleaned.csv"))
    args = p.parse_args()
    main(args.input, args.output)