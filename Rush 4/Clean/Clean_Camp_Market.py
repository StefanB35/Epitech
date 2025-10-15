import pandas as pd
import os

# Chemin du fichier CSV
df = pd.read_csv('Rush 4/Data/Camp_Market.csv', sep=';')

# Afficher les premières lignes
print("Aperçu des données :")
print(df.head())

# Infos générales
print("\nInfos générales :")
print(df.info())

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(df.describe())

# Supprimer les lignes entièrement vides
df = df.dropna(how='all')

# Supprimer les personnes qui n'ont pas de Income
df = df.dropna(subset=['Income'])

# Supprimer la colonne 'Z_CostContact' et 'Z_Revenue' si elles existent
df = df.drop(columns=['Z_CostContact'], errors='ignore')
df = df.drop(columns=['Z_Revenue'], errors='ignore') 

# Supprimer les doublons
df = df.drop_duplicates()

# Convertir la colonne 'Dt_Customer' du format YYYY-MM-DD vers DD/MM/YYYY
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%Y-%m-%d').dt.strftime('%d/%m/%Y')

# Créer le dossier Cleaned_data s'il n'existe pas
os.makedirs('Rush 4/Cleaned_data', exist_ok=True)

# Sauvegarder le fichier nettoyé
df.to_csv('Rush 4/Cleaned_data/Clean_Camp_Market.csv', index=False)