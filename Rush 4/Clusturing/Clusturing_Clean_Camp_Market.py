from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('Rush 4/Cleaned_data/Clean_Camp_Market.csv')

# Colonnes à utiliser (hors ID)
cols = [
	'Year_Birth','Education','Marital_Status','Income','Kidhome','Teenhome','Recency',
	'MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds',
	'NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth',
	'AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Response'
]

# Encodage des variables catégorielles
cat_cols = ['Education','Marital_Status']
for col in cat_cols:
	df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Préparation des données
X = df[cols]
X = X.fillna(X.mean())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Détermination du nombre optimal de clusters (méthode du coude)
inertia = []
K_range = range(1, 11)
for k in K_range:
	kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
	kmeans.fit(X_scaled)
	inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour choisir K')
plt.show()

# Appliquer KMeans avec 2 clusters
kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans_2.fit_predict(X_scaled)

# Visualisation en 2D avec PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.title('Visualisation des clusters (K=2) avec PCA')
plt.colorbar(label='Cluster')
plt.show()

# Cercle de corrélation pour les deux premières composantes principales


# Calcul des composantes principales
pcs = pca.components_

# Création du cercle
plt.figure(figsize=(8, 8))
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
plt.gca().add_artist(circle)

# Affichage des vecteurs des variables
for i, col in enumerate(cols):
	plt.arrow(0, 0, pcs[0, i], pcs[1, i], color='r', alpha=0.5, head_width=0.03)
	plt.text(pcs[0, i]*1.1, pcs[1, i]*1.1, col, fontsize=9)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Cercle de corrélation (PCA, 2 composantes)')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.grid()
plt.show()

# Calculer la moyenne des variables pour chaque cluster
cluster_means = pd.DataFrame(X_scaled, columns=cols)
cluster_means['Cluster'] = clusters
means = cluster_means.groupby('Cluster').mean()

# Préparer les données pour le radar chart
labels = cols
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # boucle

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

for idx, row in means.iterrows():
	values = row.tolist()
	values += values[:1]  # boucle
	ax.plot(angles, values, label=f'Cluster {idx}')
	ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=9)
ax.set_title("Radar chart des moyennes normalisées par cluster", y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.show()

# Polar bar charts pour chaque cluster

for idx, row in means.iterrows():
	values = row.values
	angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
	fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
	bars = ax.bar(angles, values, width=2*np.pi/num_vars, alpha=0.7)
	ax.set_xticks(angles)
	ax.set_xticklabels(labels, fontsize=9)
	ax.set_title(f'Polar Bar Chart - Cluster {idx}', y=1.08)
	plt.tight_layout()
	plt.show()