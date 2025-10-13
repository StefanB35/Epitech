import pandas as pd
import os

# Chemin du fichier CSV
df = pd.read_csv('Rush 3/Data/Credit_Data_Fichier_Clients.csv')

# Supprimer les lignes entièrement vides
df = df.dropna(how='all')

# Supprimer les doublons
df = df.drop_duplicates()

# Supprimer la collonne 'bad_client_target' si elle existe
if 'bad_client_target' in df.columns:
    df = df.drop(columns=['bad_client_target'])

# Valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# Créer le dossier Cleaned_data s'il n'existe pas
os.makedirs('Rush 3/Cleaned_data', exist_ok=True)

# Sauvegarder le fichier nettoyé
df.to_csv('Rush 3/Cleaned_data/Clean_Credit_Data_Fichier_Clients.csv', index=False)