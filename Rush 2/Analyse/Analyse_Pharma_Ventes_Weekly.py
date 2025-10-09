import pandas as pd

# Charger le fichier nettoyé
df = pd.read_csv('Rush 2/Cleaned_data/Pharma_Ventes_Weekly_Clean.csv')

# Afficher les premières lignes
print("Aperçu des données :")
print(df.head())

# Infos générales
print("\nInfos générales :")
print(df.info())

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(df.describe())

# Valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())