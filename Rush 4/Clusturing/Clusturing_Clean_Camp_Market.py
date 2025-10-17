from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

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

# Nombre de clusters souhaité (modifiable)
n_clusters = 3

# Détermination du nombre optimal de clusters (Score de silhouette)
silhouette_scores = []
for k in range(2, 11):
	kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
	labels = kmeans.fit_predict(X_scaled)
	score = silhouette_score(X_scaled, labels)
	silhouette_scores.append(score)

plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Score de silhouette')
plt.title('Score de silhouette pour choisir K')
plt.show()

# Détermination du nombre optimal de clusters (méthode du coude)
inertia = []
K_range = range(1, 11)
for k in K_range:
	kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, init='k-means++')
	kmeans.fit(X_scaled)
	inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour choisir K')
plt.show()

# Appliquer KMeans avec variable n_clusters (k-means++)
kmeans_3 = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, init='k-means++')
clusters = kmeans_3.fit_predict(X_scaled)

# Ajouter la colonne Cluster au DataFrame original (non scalé) et exporter
df['Cluster'] = clusters
output_path = 'Rush 4/Cleaned_data/Clean_Camp_Market_with_clusters.csv'
df.to_csv(output_path, index=False)
print(f"Fichier exporté avec la colonne 'Cluster' : {output_path}")

# Visualisation en 2D avec PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
# Palette fixe pour 0,1,2 puis couleurs neutres pour les éventuels clusters supplémentaires
base_colors = ['orange', 'green', 'blue']
if n_clusters <= 3:
	colors_list = base_colors[:n_clusters]
else:
	colors_list = base_colors + ['gray'] * (n_clusters - 3)

cmap = ListedColormap(colors_list)
norm = BoundaryNorm(np.arange(-0.5, n_clusters + 0.5), n_clusters)
sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap=cmap, norm=norm, alpha=0.6)
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.title(f'Visualisation des clusters (K={n_clusters}) avec PCA')
cbar = plt.colorbar(sc, ticks=range(n_clusters))
cbar.set_label('Cluster')
plt.show()

# Calcul des composantes principales
pcs = pca.components_

# Calculer la moyenne des variables pour chaque cluster (nécessaire pour colorer les vecteurs)
cluster_means = pd.DataFrame(X_scaled, columns=cols)
cluster_means['Cluster'] = clusters
means = cluster_means.groupby('Cluster').mean()

# Création du cercle
plt.figure(figsize=(8, 8))
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)
circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--')
plt.gca().add_artist(circle)

# Affichage des vecteurs des variables colorés selon le cluster où la variable a la moyenne la plus élevée
vector_color_map = {0: 'orange', 1: 'green', 2: 'blue'}
for i, col in enumerate(cols):
	# déterminer le cluster dominant pour cette variable (si présent dans means)
	try:
		dominant_cluster = int(means[col].idxmax())
	except Exception:
		dominant_cluster = None
	color = vector_color_map.get(dominant_cluster, 'gray')
	plt.arrow(0, 0, pcs[0, i], pcs[1, i], color=color, alpha=0.8, head_width=0.03)
	plt.text(pcs[0, i]*1.1, pcs[1, i]*1.1, col, fontsize=9, color=color)

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
	# Choisir la couleur selon l'index du cluster (0->orange,1->green,2->blue)
	color = base_colors[idx] if idx < len(base_colors) else 'gray'
	cluster_label = f'Cluster {idx} - {cluster_name_map.get(idx, idx)}' if 'cluster_name_map' in globals() else f'Cluster {idx}'
	ax.plot(angles, values, label=cluster_label, color=color)
	ax.fill(angles, values, color=color, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=9)
ax.set_title("Radar chart des moyennes normalisées par cluster", y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.show()

# heatmap des moyennes par cluster (valeurs normalisées entre -1 et 1)
means_norm_m1_1 = 2 * (means - means.min()) / (means.max() - means.min()) - 1
plt.figure(figsize=(12, 6))
sns.heatmap(means_norm_m1_1, annot=True, cmap='coolwarm', xticklabels=labels, vmin=-1, vmax=1)
plt.title('Heatmap normalisée (-1 à 1) des moyennes par cluster')
plt.ylabel('Cluster')
plt.xlabel('Variables')
plt.tight_layout()
plt.show()

# Compter le nombre de personnes par cluster
counts = df['Cluster'].value_counts().sort_index()
counts_df = counts.rename_axis('Cluster').reset_index(name='Count')

print("Nombre de personnes par cluster :")
print(counts_df.to_string(index=False))

# Exporter les résultats
counts_output = 'Rush 4/Cleaned_data/cluster_counts.csv'
counts_df.to_csv(counts_output, index=False)
print(f"Fichier exporté : {counts_output}")

# Visualisation simple (bar chart)
plt.figure(figsize=(6, 4))
sns.barplot(data=counts_df, x='Cluster', y='Count', palette='viridis')
plt.title('Nombre de personnes par cluster')
plt.tight_layout()
plt.show()