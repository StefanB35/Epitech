import pandas as pd
import os

# Chemin du fichier CSV
df = pd.read_csv('Rush 2/Data/Pharma_Ventes_hourly.csv')

# Supprimer les lignes entièrement vides
df = df.dropna(how='all')

# Supprimer les doublons
df = df.drop_duplicates()

# Séparer la colonne 'datum' en date et heure
df[['date', 'heure']] = df['datum'].str.split(' ', expand=True)

# Convertir la colonne 'date' au format JJ/MM/YYYY
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y').dt.strftime('%d/%m/%Y')

# Supprimer la colonne 'datum' & 'Hour'
df = df.drop(columns=['datum'])
df = df.drop(columns=['Hour'])

# Créer le dossier Cleaned_data s'il n'existe pas
os.makedirs('Rush 2/Cleaned_data', exist_ok=True)

# Sauvegarder le fichier nettoyé
df.to_csv('Rush 2/Cleaned_data/Pharma_Ventes_Hourly_Clean.csv', index=False)