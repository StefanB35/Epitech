import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le fichier nettoyé
df = pd.read_csv('Rush 2/Cleaned_data/Pharma_Ventes_Hourly_Clean.csv')

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

###################
# Analyse poussée #
###################

# Groupes de molécules
groupes = {
	'M': ['M01AB', 'M01AE'],
	'N': ['N02BA', 'N02BE', 'N05B', 'N05C'],
	'R': ['R03', 'R06']
}

# 1. Pour chaque groupe, afficher une heatmap de la somme des ventes du groupe
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
for cat, cols in groupes.items():
	df[cat + '_sum'] = df[cols].sum(axis=1)
	pivot = df.pivot_table(index='Hour', columns='Weekday Name', values=cat + '_sum', aggfunc='mean')
	pivot = pivot[weekday_order]
	plt.figure(figsize=(10, 6))
	sns.heatmap(pivot, cmap='YlGnBu', annot=False)
	plt.title(f'Heatmap des ventes catégorie {cat} (somme) par heure et jour de la semaine')
	plt.xlabel('Jour de la semaine')
	plt.ylabel('Heure')
	plt.tight_layout()
	plt.show()