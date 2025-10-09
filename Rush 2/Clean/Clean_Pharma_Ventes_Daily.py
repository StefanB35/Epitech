import pandas as pd
import os

# Chemin du fichier CSV
df = pd.read_csv('Rush 2/Data/Pharma_Ventes_Daily.csv')

# Supprimer les lignes entièrement vides
df = df.dropna(how='all')

# Supprimer les doublons
df = df.drop_duplicates()

# Convertir la colonne 'datum' au format JJ/MM/YYYY
df['datum'] = pd.to_datetime(df['datum'], format='%m/%d/%Y').dt.strftime('%d/%m/%Y')

# Renommer la colonne 'datum' en 'date'
df = df.rename(columns={'datum': 'date'})

# Supprimer la colonne 'Hour'
df = df.drop(columns=['Hour'])

# Créer le dossier Cleaned_data s'il n'existe pas
os.makedirs('Rush 2/Cleaned_data', exist_ok=True)

# Sauvegarder le fichier nettoyé
df.to_csv('Rush 2/Cleaned_data/Pharma_Ventes_Daily_Clean.csv', index=False)