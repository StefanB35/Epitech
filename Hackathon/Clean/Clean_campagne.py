import os
import re
import sys
from pathlib import Path
import pandas as pd

# Clean_campagne.py
# Fusionne plusieurs fichiers CSV
# Usage: python Clean_campagne.py [output.csv]


FILES = [
    r"Hackathon\Data\Vaccin\2021\campagne-2021.csv",
    r"Hackathon\Data\Vaccin\2022\campagne-2022.csv",
    r"Hackathon\Data\Vaccin\2023\campagne-2023.csv",
    r"Hackathon\Data\Vaccin\2024\campagne-2024.csv",
]


def read_csv_guess_sep(path: Path) -> pd.DataFrame:
    # Use pandas sep=None with python engine to let it sniff the delimiter
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # fallback tries
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep)
                break
            except Exception:
                df = None
        if df is None:
            raise
    return df


def extract_year_from_name(path: Path) -> str:
    m = re.search(r"(\d{4})", path.name)
    return m.group(1) if m else ""


def main(output: str = None):
    dfs = []
    existing = []
    for p in FILES:
        path = Path(p)
        if not path.exists():
            print(f"Warning: '{path}' not found, skipping.")
            continue
        df = read_csv_guess_sep(path)
        dfs.append(df)
        existing.append(path)

    if not dfs:
        print("No files found to merge. Exiting.")
        return

    merged = pd.concat(dfs, ignore_index=True, sort=False)

    # default output: Hackathon\Data_Clean folder
    if output is None:
        out_dir = Path("Hackathon/Data_Clean")
        out_dir.mkdir(parents=True, exist_ok=True)  # créer le dossier si nécessaire
        output = out_dir / "campagne-merged.csv"
    else:
        output = Path(output)

    merged.to_csv(output, index=False, encoding="utf-8")
    print(f"Merged {len(dfs)} files -> {output}")


if __name__ == "__main__":
    out_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(out_arg)