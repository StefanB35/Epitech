#!/usr/bin/env python3
"""
Clean_Taux_infermier.py - Nettoyage des données de taux d'infirmiers

DESCRIPTION:
    Script de conversion et nettoyage des données Excel de taux d'infirmiers.
    Convertit le fichier Excel en CSV et effectue des transformations spécifiques.

FONCTIONNALITÉS:
    - Conversion Excel vers CSV
    - Extraction des codes régions depuis la colonne indicateur
    - Suppression de la colonne "Auxiliaires médicaux"
    - Création d'une colonne 'region' basée sur l'extraction
    - Nettoyage et normalisation des données

TRANSFORMATIONS SPÉCIFIQUES:
    - Extraction du code région depuis la colonne "indicateur"
    - Suppression des colonnes contenant "auxiliaires" ou "médicaux"
    - Normalisation des noms de colonnes

USAGE:
    python Clean_Taux_infermier.py

AUTEUR: Stéfan Beaulieu
DATE: 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================
import sys
from pathlib import Path
import pandas as pd
import re

# =============================================================================
# FONCTIONS DE TRANSFORMATION
# =============================================================================

def transform_dataframe(df):
    """
    Transforme le DataFrame pour extraire codes régions et nettoyer les données.
    
    Args:
        df (pd.DataFrame): DataFrame original à transformer
        
    Returns:
        pd.DataFrame: DataFrame transformé et nettoyé
    """
    # Copier le DataFrame pour éviter les modifications sur l'original
    df_clean = df.copy()
    
    # Supprimer la colonne "Auxiliaires médicaux" si elle existe
    columns_to_drop = []
    for col in df_clean.columns:
        if 'auxiliaires' in col.lower() or 'médicaux' in col.lower():
            columns_to_drop.append(col)
    
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
        print(f"Colonnes supprimées: {columns_to_drop}")
    
    # Chercher la colonne qui contient les informations de région
    region_col = None
    for col in df_clean.columns:
        if 'region' in col.lower() or 'insee' in col.lower():
            region_col = col
            break
    
    if region_col:
        # Extraire le code région (les chiffres au début)
        df_clean['code_region'] = df_clean[region_col].astype(str).str.extract(r'^(\d+)')[0]
        
        # Extraire le nom de région (après le tiret)
        df_clean['nom_region'] = df_clean[region_col].astype(str).str.replace(r'^\d+-?\s*', '', regex=True)
        
        # Nettoyer les espaces
        df_clean['nom_region'] = df_clean['nom_region'].str.strip()
        
        # Supprimer la colonne originale REGION INSEE maintenant qu'on a extrait les infos
        df_clean = df_clean.drop(columns=[region_col])
        
        print(f"Codes et noms de régions extraits de la colonne: {region_col}")
        print(f"Colonne {region_col} supprimée")
    
    # Nettoyer les nombres (enlever virgules comme séparateurs de milliers)
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Essayer de convertir en numérique si ça ressemble à un nombre
            sample = df_clean[col].dropna().astype(str).iloc[0] if not df_clean[col].dropna().empty else ""
            if re.search(r'[\d,\.]+', sample):
                df_clean[col] = df_clean[col].astype(str).str.replace(',', '', regex=False)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
    
    return df_clean

def main():
    # Chemin vers le fichier Excel source
    xlsx_path = Path("Hackathon") / "Data" / "Taux_infermier.xlsx"
    if not xlsx_path.exists():
        print(f"Fichier introuvable: {xlsx_path}")
        sys.exit(1)

    # Dossier de sortie dans Data_Clean
    output_dir = Path("Hackathon") / "Data_Clean"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lire toutes les feuilles pour gérer le cas de plusieurs feuilles
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    if len(sheets) == 1:
        # une seule feuille -> nom simple
        df = next(iter(sheets.values()))
        df_transformed = transform_dataframe(df)
        csv_path = output_dir / "Taux_infermier_cleaned.csv"
        df_transformed.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Exporté dans Data_Clean: {csv_path}")
        print(f"Forme finale: {df_transformed.shape}")
        print(f"Colonnes: {list(df_transformed.columns)}")
    else:
        # plusieurs feuilles -> un fichier par feuille
        for name, df in sheets.items():
            df_transformed = transform_dataframe(df)
            safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
            csv_path = output_dir / f"Taux_infermier_{safe_name}_cleaned.csv"
            df_transformed.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"Exporté dans Data_Clean: {csv_path}")
            print(f"Forme finale: {df_transformed.shape}")

if __name__ == "__main__":
    main()