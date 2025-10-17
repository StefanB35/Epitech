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

# Remplacer 'Alone' par 'Single' dans la colonne 'Marital_Status'
df['Marital_Status'] = df['Marital_Status'].replace('Alone', 'Single')

# Supprimer les personnes qui n'ont pas de Income
df = df.dropna(subset=['Income'])

# Supprimer les personnes avec 'Absurd' ou 'YOLO' dans 'Marital_Status'
df = df[~df['Marital_Status'].isin(['Absurd', 'YOLO'])]

# Supprimer la colonne 'Z_CostContact' et 'Z_Revenue' si elles existent
df = df.drop(columns=['Z_CostContact'], errors='ignore')
df = df.drop(columns=['Z_Revenue'], errors='ignore') 

# Supprimer les personnes de plus de 90 ans selon Year_Birth
current_year = 2022
df = df[(current_year - df['Year_Birth']) <= 90]

# Supprimer les doublons
df = df.drop_duplicates()

# Convertir la colonne 'Dt_Customer' du format YYYY-MM-DD vers DD/MM/YYYY
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%Y-%m-%d').dt.strftime('%d/%m/%Y')

# Créer le dossier Cleaned_data s'il n'existe pas
os.makedirs('Rush 4/Cleaned_data', exist_ok=True)

# Sauvegarder le fichier nettoyé
df.to_csv('Rush 4/Cleaned_data/Clean_Camp_Market.csv', index=False)