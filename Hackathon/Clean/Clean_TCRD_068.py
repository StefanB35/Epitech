from pathlib import Path
import re
import pandas as pd
import numpy as np
import sys

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Script: Clean_TCRD_068.py
# Objectif: nettoyer Hackathon/Data/TCRD_068.xlsx et écrire un fichier nettoyé dans Data_Clean
# Usage: python Clean_TCRD_068.py


INPUT_PATH = Path("Hackathon") / "Data" / "TCRD_068.xlsx"
OUTPUT_DIR = Path("Hackathon") / "Data_Clean"
OUTPUT_PATH = OUTPUT_DIR / "TCRD_068_cleaned.csv"  # Changed to CSV for easier handling

# Définir les noms de colonnes attendus
EXPECTED_COLUMNS = [
    "code",
    "region", 
    "ensemble_des_medecins",
    "ensemble_des_medecins_densite_pour_100_000_habitants",
    "dont_generalistes_densite_pour_100_000_habitants", 
    "dont_specialistes_densite_pour_100_000_habitants",
    "chirurg_dentistes_densite_pour_100_000_habitants",
    "pharm_densite_pour_100_000_habitants"
]

def normalize_columns(cols):
    # normaliser noms de colonnes selon le mapping défini
    original_names = [
        "code", "région", "Ensemble des médecins",
        "Ensemble des médecins Densité pour 100 000 habitants",
        "dont généralistes Densité pour 100 000 habitants", 
        "dont spécialistes Densité pour 100 000 habitants",
        "Chirurg. dentistes Densité pour 100 000 habitants",
        "Pharm. Densité pour 100 000 habitants"
    ]
    
    # Si on a exactement le bon nombre de colonnes, on utilise nos noms
    if len(cols) == len(EXPECTED_COLUMNS):
        return EXPECTED_COLUMNS
    
    # Sinon, normalisation standard
    new = []
    for c in cols:
        s = str(c).strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^\w_]", "", s)
        new.append(s if s else "col")
    # garantir unicité
    seen = {}
    uniq = []
    for name in new:
        n = name
        n = name
        i = 1
        while n in seen:
            i += 1
            n = f"{name}_{i}"
        seen[n] = True
        uniq.append(n)
    return uniq

def try_numeric_conversion(series):
    # tente conversion numérique; si plus de 50% non-null après conversion on garde
    coerced = pd.to_numeric(series.astype(str).replace({"nan": np.nan, "None": np.nan}), errors="coerce")
    non_null_ratio = coerced.notna().sum() / max(1, len(coerced))
    return coerced if non_null_ratio >= 0.5 else series

def try_datetime_conversion(series):
    coerced = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    non_null_ratio = coerced.notna().sum() / max(1, len(coerced))
    return coerced if non_null_ratio >= 0.5 else series

def clean_dataframe(df):
    # Sauvegarder forme initiale
    initial_shape = df.shape

    # Supprimer colonnes entièrement vides
    df = df.dropna(axis=1, how="all")

    # Supprimer lignes entièrement vides
    df = df.dropna(axis=0, how="all")

    # Normaliser en-têtes
    df.columns = normalize_columns(df.columns)

    # Nettoyer chaînes: trim et collapse d'espaces
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].map(lambda x: " ".join(str(x).split()) if pd.notna(x) else x)

    # Tenter conversions numériques et dates colonne par colonne
    # SAUF pour la colonne 'code' qui doit rester en string
    # ET forcer certaines colonnes en int
    int_columns = [
        'ensemble_des_medecins',
        'ensemble_des_medecins_densite_pour_100_000_habitants',
        'dont_generalistes_densite_pour_100_000_habitants',
        'dont_specialistes_densite_pour_100_000_habitants',
        'chirurg_dentistes_densite_pour_100_000_habitants',
        'pharm_densite_pour_100_000_habitants'
    ]
    
    for c in df.columns:
        ser = df[c]
        # Exclure la colonne 'code' des conversions automatiques
        if c.lower() == 'code':
            # S'assurer que la colonne code reste en string
            df[c] = ser.astype(str)
            continue
        
        # Forcer certaines colonnes en int
        if c in int_columns:
            # Convertir en numérique puis en int, en gérant les NaN
            numeric_ser = pd.to_numeric(ser, errors='coerce')
            df[c] = numeric_ser.fillna(0).astype(int)
            continue
            
        # si dtype object, essayer numérique puis datetime
        if ser.dtype == object:
            new_ser = try_numeric_conversion(ser)
            if new_ser is ser:
                new_ser = try_datetime_conversion(ser)
            df[c] = new_ser

    # Retirer colonnes devenues entièrement vides après conversions
    df = df.dropna(axis=1, how="all")

    # Enlever doublons
    before = len(df)
    df = df.drop_duplicates()
    dup_removed = before - len(df)

    final_shape = df.shape
    return df, initial_shape, final_shape, dup_removed

def main():
    if not INPUT_PATH.exists():
        print(f"Fichier introuvable: {INPUT_PATH}", file=sys.stderr)
        sys.exit(2)

    # Lire toutes les feuilles
    xls = pd.read_excel(INPUT_PATH, sheet_name=None)

    # Créer le dossier de sortie s'il n'existe pas
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pour chaque feuille, on prend la première qui a les bonnes colonnes
    main_df = None
    for name, df in xls.items():
        if len(df.columns) == len(EXPECTED_COLUMNS):
            cdf, init_shape, final_shape, dup_removed = clean_dataframe(df)
            main_df = cdf
            print(f"Feuille utilisée: {name}  lignes: {init_shape[0]}->{final_shape[0]}  colonnes: {init_shape[1]}->{final_shape[1]}  doublons supprimés: {dup_removed}")
            break
    
    if main_df is None:
        print("Aucune feuille avec le bon nombre de colonnes trouvée")
        sys.exit(1)

    # Écrire fichier nettoyé en CSV
    main_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Fichier nettoyé écrit: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()