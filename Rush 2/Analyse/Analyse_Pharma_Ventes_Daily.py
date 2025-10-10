import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le fichier nettoyé
df = pd.read_csv('Rush 2/Cleaned_data/Pharma_Ventes_Daily_Clean.csv')

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

# Liste des molécules
molecules = ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']

# Groupes de molécules
groupes = {
    'M': ['M01AB', 'M01AE'],
    'N': ['N02BA', 'N02BE', 'N05B', 'N05C'],
    'R': ['R03', 'R06']
}

# 1. Évolution des ventes par jour pour chaque produit
plt.figure(figsize=(14, 7))

for mol in molecules:
    plt.plot(df['date'], df[mol], label=mol)

plt.title("Évolution des ventes par jour pour chaque produit")
plt.xlabel("Date")
plt.ylabel("Ventes")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Stacked area chart : part de marché des catégories dans le temps
for cat, cols in groupes.items():
    df[cat + '_sum'] = df[cols].sum(axis=1)

df_area = df[['date', 'M_sum', 'N_sum', 'R_sum']].copy()
df_area['date'] = pd.to_datetime(df_area['date'], format='%d/%m/%Y')
df_area = df_area.sort_values('date')
df_area.set_index('date', inplace=True)

plt.figure(figsize=(14, 7))
plt.stackplot(df_area.index, df_area['M_sum'], df_area['N_sum'], df_area['R_sum'], labels=['M', 'N', 'R'], alpha=0.8)
plt.title('Part de marché des catégories dans le temps (stacked area chart)')
plt.xlabel('Date')
plt.ylabel('Ventes (somme des molécules)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# 3.1 Moyenne par nom de jour de la semaine
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_grouped = df.groupby('Weekday Name')[molecules].mean().reindex(weekday_order)

plt.figure(figsize=(12, 6))
for mol in molecules:
    plt.plot(df_grouped.index, df_grouped[mol], marker='o', label=mol)

plt.title("Ventes moyennes par jour de la semaine pour chaque molécule")
plt.xlabel("Jour de la semaine")
plt.ylabel("Ventes moyennes")
plt.legend()
plt.tight_layout()
plt.show()

# 3.2 Moyenne par nom de jour de la semaine (par catégorie)
for cat, cols in groupes.items():
    df[cat + '_sum'] = df[cols].sum(axis=1)

df_grouped_cat = df.groupby('Weekday Name')[['M_sum', 'N_sum', 'R_sum']].mean().reindex(weekday_order)

plt.figure(figsize=(12, 6))
for cat in ['M', 'N', 'R']:
    plt.plot(df_grouped_cat.index, df_grouped_cat[cat+'_sum'], marker='o', label=f'Catégorie {cat}')

plt.title("Ventes moyennes par jour de la semaine pour chaque catégorie")
plt.xlabel("Jour de la semaine")
plt.ylabel("Ventes moyennes (somme)")
plt.legend()
plt.tight_layout()
plt.show()

# 4.1 Moyenne par mois et par molécule
df['Month'] = df['Month'].astype(int)
df['Year'] = df['Year'].astype(int)
df_monthly = df.groupby(['Year', 'Month'])[molecules].mean().reset_index()

df_monthly['date'] = pd.to_datetime(df_monthly['Year'].astype(str) + '-' + df_monthly['Month'].astype(str) + '-01')

plt.figure(figsize=(14, 7))
for mol in molecules:
    plt.plot(df_monthly['date'], df_monthly[mol], marker='o', label=mol)

plt.title("Ventes moyennes mensuelles par molécule")
plt.xlabel("Date")
plt.ylabel("Ventes moyennes")
plt.legend()
plt.tight_layout()
plt.show()

# 4.2 Moyenne par mois et par catégorie
df_monthly_cat = df.groupby(['Year', 'Month'])[['M_sum', 'N_sum', 'R_sum']].mean().reset_index()
df_monthly_cat['date'] = pd.to_datetime(df_monthly_cat['Year'].astype(str) + '-' + df_monthly_cat['Month'].astype(str) + '-01')

plt.figure(figsize=(14, 7))
for cat in ['M', 'N', 'R']:
    plt.plot(df_monthly_cat['date'], df_monthly_cat[cat+'_sum'], marker='o', label=f'Catégorie {cat}')

plt.title("Ventes moyennes mensuelles par catégorie")
plt.xlabel("Date")
plt.ylabel("Ventes moyennes (somme)")
plt.legend()
plt.tight_layout()
plt.show()

# 5. Box plot : dispersion des ventes par mois (par catégorie)
for cat, cols in groupes.items():
    df[cat + '_sum'] = df[cols].sum(axis=1)

plt.figure(figsize=(12, 6))
for cat in groupes.keys():
    sns.boxplot(x='Month', y=cat+'_sum', data=df, showfliers=False)
    plt.title(f'Dispersion des ventes par mois (catégorie {cat})')
    plt.xlabel('Mois')
    plt.ylabel('Ventes (somme)')
    plt.tight_layout()
    plt.show()
