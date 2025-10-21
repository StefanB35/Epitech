from pathlib import Path
import argparse
import logging
import re
import unicodedata
import pandas as pd
import numpy as np

#!/usr/bin/env python3
# GitHub Copilot
"""
Clean Hackathon/Data/finalPredDoses.csv and write a cleaned version.
Usage:
    python Clean_finalPredDoses.py --input "Hackathon/Data/finalPredDoses.csv" --output "Hackathon/Data/finalPredDoses.cleaned.csv"
"""




logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def normalize_colname(s: str) -> str:
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s-]", "", s)  # remove punctuation
    s = s.lower()
    s = re.sub(r"[\s-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def clean_string_value(v):
    if pd.isna(v):
        return v
    if not isinstance(v, str):
        v = str(v)
    v = v.strip()
    v = unicodedata.normalize("NFKC", v)
    # remove control characters
    v = re.sub(r"[\x00-\x1f\x7f]", "", v)
    # collapse multiple spaces
    v = re.sub(r"\s+", " ", v)
    return v


def detect_date_columns(cols):
    date_keywords = ["date", "jour", "day", "mois", "month", "semaine", "week", "année", "year", "année predite"]
    return [c for c in cols if any(k in c for k in date_keywords)]


def main(input_path: Path, output_path: Path):
    if not input_path.exists():
        logging.error("Input file does not exist: %s", input_path)
        return

    # read with heuristics for encoding and separators
    try:
        df = pd.read_csv(input_path, sep=None, engine="python", encoding="utf-8")
    except Exception:
        df = pd.read_csv(input_path, engine="python", encoding="latin1")

    # normalize column names
    original_cols = list(df.columns)
    new_cols = [normalize_colname(c) for c in original_cols]
    
    # Remove "region_" prefix from column names
    new_cols = [col.replace("region_", "") if col.startswith("region_") else col for col in new_cols]
    
    rename_map = dict(zip(original_cols, new_cols))
    df.rename(columns=rename_map, inplace=True)
    logging.info("Columns normalized and region_ prefix removed: %s", new_cols)

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
    
    # Remove the first 14 columns since we now have the region info in 'region' column
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
    
    # region column should also be string
    if 'region' in df.columns:
        df['region'] = df['region'].astype(str)
        logging.info("Forced column 'region' to string type")
    
    # annee, annee_predite and code columns should be integers
    int_cols = ['annee', 'annee_predite', 'code', 'actes', 'doses']
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
    skip_cols = region_and_age_cols + int_cols + date_cols + ['region']
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
    p = argparse.ArgumentParser(description="Clean finalPredDoses.csv")
    p.add_argument("--input", "-i", type=Path, default=Path("Hackathon/Data/finalPredDoses.csv"))
    p.add_argument("--output", "-o", type=Path, default=Path("Hackathon/Data_Clean/finalPredDoses_cleaned.csv"))
    args = p.parse_args()
    main(args.input, args.output)