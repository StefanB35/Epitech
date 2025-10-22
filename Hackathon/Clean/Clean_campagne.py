#!/usr/bin/env python3
"""
Clean_campagne.py - Script de fusion de données de campagnes de vaccination

DESCRIPTION:
    Ce script fusionne plusieurs fichiers CSV contenant les données des campagnes 
    de vaccination de différentes années (2021-2024) en un seul fichier consolidé.
    
    Le script utilise une détection automatique du séparateur CSV et gère les 
    encodages multiples pour assurer une lecture correcte des données.

USAGE:
    python Clean_campagne.py [fichier_sortie.csv]
    
    Paramètres:
        fichier_sortie (optionnel): Nom du fichier de sortie. 
                                   Par défaut: Hackathon/Data_Clean/campagne-merged.csv

EXEMPLE:
    python Clean_campagne.py                           # Utilise la sortie par défaut
    python Clean_campagne.py "mon_fichier_merge.csv"  # Sortie personnalisée

AUTEUR: Stéfan Beaulieu
DATE: 2025
"""

# Import des bibliothèques nécessaires
import os
import re
import sys
from pathlib import Path
import pandas as pd


# =============================================================================
# CONFIGURATION - Fichiers d'entrée
# =============================================================================

# Liste des fichiers CSV de campagnes de vaccination à fusionner
# Chaque fichier correspond à une année de campagne (2021-2024)
FILES = [
    r"Hackathon\Data\Vaccin\2021\campagne-2021.csv",  # Données campagne 2021
    r"Hackathon\Data\Vaccin\2022\campagne-2022.csv",  # Données campagne 2022
    r"Hackathon\Data\Vaccin\2023\campagne-2023.csv",  # Données campagne 2023
    r"Hackathon\Data\Vaccin\2024\campagne-2024.csv",  # Données campagne 2024
]


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def read_csv_guess_sep(path: Path) -> pd.DataFrame:
    """
    Lit un fichier CSV avec détection automatique du séparateur.
    
    Cette fonction tente de détecter automatiquement le séparateur utilisé
    dans le fichier CSV (virgule, point-virgule, tabulation) pour assurer
    une lecture correcte même si les formats varient entre les fichiers.
    
    Args:
        path (Path): Chemin vers le fichier CSV à lire
        
    Returns:
        pd.DataFrame: DataFrame contenant les données du fichier CSV
        
    Raises:
        Exception: Si aucun séparateur valide n'est trouvé
        
    Note:
        - Utilise d'abord pandas avec sep=None pour la détection automatique
        - En cas d'échec, teste manuellement les séparateurs courants
    """
    try:
        # Première tentative : détection automatique avec pandas
        df = pd.read_csv(path, sep=None, engine="python")
        print(f"Fichier lu avec détection automatique: {path.name}")
    except Exception:
        # Tentatives de fallback avec des séparateurs spécifiques
        print(f"Détection automatique échouée pour {path.name}, test manuel...")
        df = None
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep)
                print(f"✓ Fichier lu avec séparateur '{sep}': {path.name}")
                break
            except Exception:
                continue
        
        # Si aucun séparateur ne fonctionne, lever une exception
        if df is None:
            raise Exception(f"Impossible de lire le fichier {path.name}")
    
    return df


def extract_year_from_name(path: Path) -> str:
    """
    Extrait l'année du nom de fichier.
    
    Utilise une expression régulière pour trouver une séquence de 4 chiffres
    dans le nom du fichier, qui correspond typiquement à l'année.
    
    Args:
        path (Path): Chemin du fichier dont extraire l'année
        
    Returns:
        str: Année extraite sous forme de chaîne, ou chaîne vide si non trouvée
        
    Example:
        extract_year_from_name(Path("campagne-2021.csv")) -> "2021"
    """
    # Recherche d'un pattern de 4 chiffres consécutifs dans le nom du fichier
    match = re.search(r"(\d{4})", path.name)
    return match.group(1) if match else ""


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main(output: str = None):
    """
    Fonction principale qui orchestre la fusion des fichiers CSV de campagnes.
    
    Cette fonction:
    1. Lit tous les fichiers CSV spécifiés dans FILES
    2. Les fusionne en un seul DataFrame
    3. Sauvegarde le résultat dans le fichier de sortie spécifié
    
    Args:
        output (str, optionnel): Chemin du fichier de sortie. 
                                Si None, utilise "Hackathon/Data_Clean/campagne-merged.csv"
    
    Process:
        - Vérification de l'existence de chaque fichier
        - Lecture avec détection automatique du séparateur
        - Fusion de tous les DataFrames
        - Création du dossier de sortie si nécessaire
        - Sauvegarde au format CSV avec encodage UTF-8
    """
    print("=" * 60)
    print("FUSION DES FICHIERS DE CAMPAGNES DE VACCINATION")
    print("=" * 60)
    
    # Listes pour stocker les DataFrames et les chemins des fichiers existants
    dfs = []           # DataFrames lus avec succès
    existing = []      # Chemins des fichiers qui existent
    
    # Parcourir tous les fichiers spécifiés dans la configuration
    print(f"Traitement de {len(FILES)} fichiers...")
    
    for i, file_path in enumerate(FILES, 1):
        path = Path(file_path)
        print(f"\n[{i}/{len(FILES)}] Traitement: {path.name}")
        
        # Vérifier si le fichier existe
        if not path.exists():
            print(f"Fichier non trouvé: '{path}', ignoré.")
            continue
        
        try:
            # Lire le fichier CSV avec détection automatique du séparateur
            df = read_csv_guess_sep(path)
            print(f"Lignes lues: {len(df)}, Colonnes: {len(df.columns)}")
            
            # Ajouter aux listes des fichiers traités
            dfs.append(df)
            existing.append(path)
            
        except Exception as e:
            print(f"Erreur lors de la lecture de {path.name}: {e}")
            continue
    
    # Vérifier qu'au moins un fichier a été lu avec succès
    if not dfs:
        print("\nAucun fichier trouvé ou lisible. Arrêt du processus.")
        return
    
    print(f"\n{len(dfs)} fichiers lus avec succès")
    
    # Fusionner tous les DataFrames en un seul
    print("\nFusion des données en cours...")
    merged = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"✓ Fusion terminée: {len(merged)} lignes totales, {len(merged.columns)} colonnes")
    
    # Déterminer le chemin de sortie
    if output is None:
        # Chemin par défaut dans le dossier Data_Clean
        out_dir = Path("Hackathon/Data_Clean")
        out_dir.mkdir(parents=True, exist_ok=True)  # Créer le dossier si nécessaire
        output = out_dir / "campagne-merged.csv"
        print(f"Dossier de sortie créé: {out_dir}")
    else:
        output = Path(output)
    
    # Sauvegarder le fichier fusionné
    print(f"\nSauvegarde vers: {output}")
    try:
        merged.to_csv(output, index=False, encoding="utf-8")
        print(f"Fusion réussie! {len(dfs)} fichiers -> {output}")
        print(f"Résultat final: {len(merged)} lignes, {len(merged.columns)} colonnes")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        return
    
    print("=" * 60)
    print("PROCESSUS TERMINÉ AVEC SUCCÈS")
    print("=" * 60)


# =============================================================================
# POINT D'ENTRÉE DU SCRIPT
# =============================================================================

if __name__ == "__main__":
    """
    Point d'entrée principal du script.
    
    Gère les arguments de ligne de commande:
    - sys.argv[1]: Fichier de sortie optionnel
    
    Usage depuis la ligne de commande:
        python Clean_campagne.py                    # Sortie par défaut
        python Clean_campagne.py "mon_fichier.csv" # Sortie personnalisée
    """
    # Récupérer l'argument de sortie depuis la ligne de commande (optionnel)
    output_argument = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Lancer le processus principal
    main(output_argument)