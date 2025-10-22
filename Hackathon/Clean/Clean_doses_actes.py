#!/usr/bin/env python3
"""
Clean_doses_actes.py - Fusion des données de doses et actes médicaux

DESCRIPTION:
    Script de fusion pour consolider les fichiers CSV de doses et actes médicaux
    de vaccination sur plusieurs années (2021-2024). Similaire à Clean_campagne.py
    mais spécialisé pour les données d'actes et doses administrées.

FONCTIONNALITÉS:
    - Fusion de fichiers CSV doses-actes de 2021 à 2024
    - Détection automatique des séparateurs CSV
    - Gestion des encodages multiples
    - Consolidation temporelle des données d'actes médicaux
    - Sauvegarde dans le dossier Data_Clean standardisé

DONNÉES TRAITÉES:
    - Doses de vaccins administrées par année
    - Actes médicaux liés à la vaccination
    - Statistiques temporelles sur 4 années

USAGE:
    python Clean_doses_actes.py [fichier_sortie.csv]

EXEMPLE:
    python Clean_doses_actes.py                           # Sortie par défaut
    python Clean_doses_actes.py "doses_actes_merged.csv" # Sortie personnalisée

AUTEUR: Stéfan Beaulieu  
DATE: 2025
"""

# =============================================================================
# IMPORTS ET CONFIGURATION
# =============================================================================
import os
import re
import sys
from pathlib import Path
import pandas as pd


# =============================================================================
# CONFIGURATION - Fichiers de doses et actes à fusionner
# =============================================================================

# Liste des fichiers CSV de doses et actes médicaux par année
# Chaque fichier contient les données d'actes de vaccination pour une année donnée
FILES = [
    r"Hackathon\Data\Vaccin\2021\doses-actes-2021.csv",  # Actes et doses 2021
    r"Hackathon\Data\Vaccin\2022\doses-actes-2022.csv",  # Actes et doses 2022
    r"Hackathon\Data\Vaccin\2023\doses-actes-2023.csv",  # Actes et doses 2023
    r"Hackathon\Data\Vaccin\2024\doses-actes-2024.csv",  # Actes et doses 2024
]

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def read_csv_guess_sep(path: Path) -> pd.DataFrame:
    """
    Lit un fichier CSV avec détection automatique du séparateur.
    Identique à la fonction utilisée dans Clean_campagne.py pour cohérence.
    
    Args:
        path (Path): Chemin vers le fichier CSV
        
    Returns:
        pd.DataFrame: Données chargées
        
    Raises:
        Exception: Si aucun format de lecture ne fonctionne
    """
    try:
        # Tentative avec détection automatique du séparateur
        df = pd.read_csv(path, sep=None, engine="python")
        print(f"✓ Fichier lu (auto-détection): {path.name}")
    except Exception:
        # Tentatives manuelles avec séparateurs courants
        df = None
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep)
                print(f"✓ Fichier lu (séparateur '{sep}'): {path.name}")
                break
            except Exception:
                continue
                continue
        
        # Si aucun séparateur ne fonctionne
        if df is None:
            raise Exception(f"Impossible de lire le fichier: {path.name}")
    
    return df


def extract_year_from_name(path: Path) -> str:
    """
    Extrait l'année du nom de fichier doses-actes.
    
    Args:
        path (Path): Chemin du fichier
        
    Returns:
        str: Année extraite ou chaîne vide si non trouvée
    """
    match = re.search(r"(\d{4})", path.name)
    return match.group(1) if match else ""


# =============================================================================
# FONCTION PRINCIPALE DE FUSION
# =============================================================================

def main(output: str = None):
    """
    Fonction principale pour fusionner les fichiers de doses et actes.
    
    Args:
        output (str, optionnel): Chemin de sortie personnalisé
    """
    print("=" * 60)
    print("FUSION DES FICHIERS DOSES-ACTES DE VACCINATION")
    print("=" * 60)
    
    # Collecte des DataFrames et fichiers traités
    dfs = []
    existing = []
    
    print(f"Traitement de {len(FILES)} fichiers d'actes médicaux...")
    
    for i, file_path in enumerate(FILES, 1):
        path = Path(file_path)
        print(f"\n[{i}/{len(FILES)}] Traitement: {path.name}")
        
        if not path.exists():
            print(f"Fichier non trouvé: '{path}', ignoré.")
            continue
        
        try:
            df = read_csv_guess_sep(path)
            print(f"Données: {len(df)} lignes, {len(df.columns)} colonnes")
            
            dfs.append(df)
            existing.append(path)
            
        except Exception as e:
            print(f"Erreur de lecture {path.name}: {e}")
            continue
    
    # Vérification qu'au moins un fichier a été lu
    if not dfs:
        print("\nAucun fichier trouvé ou lisible pour doses-actes.")
        return
    
    print(f"\n{len(dfs)} fichiers traités avec succès")
    
    # Fusion des DataFrames
    print("\nFusion des données d'actes médicaux...")
    merged = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"Fusion terminée: {len(merged)} lignes, {len(merged.columns)} colonnes")
    
    # Définition du chemin de sortie
    if output is None:
        out_dir = Path("Hackathon/Data_Clean")
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "doses-actes-merged.csv"
        print(f"Dossier de sortie: {out_dir}")
    else:
        output = Path(output)
    
    # Sauvegarde finale
    try:
        merged.to_csv(output, index=False, encoding="utf-8")
        print(f"\nFusion réussie! {len(dfs)} fichiers → {output}")
        print(f"Résultat: {len(merged)} lignes, {len(merged.columns)} colonnes")
    except Exception as e:
        print(f"Erreur de sauvegarde: {e}")
        return
    
    print("=" * 60)
    print("FUSION DOSES-ACTES TERMINÉE AVEC SUCCÈS")
    print("=" * 60)


# =============================================================================
# POINT D'ENTRÉE DU SCRIPT
# =============================================================================

if __name__ == "__main__":
    """Point d'entrée avec gestion des arguments de ligne de commande."""
    output_argument = sys.argv[1] if len(sys.argv) > 1 else None
    main(output_argument)