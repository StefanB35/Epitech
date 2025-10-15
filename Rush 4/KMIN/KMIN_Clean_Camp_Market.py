import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
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

k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

print(df[['ID','cluster']].head())

# Visualisation simple (PCA possible si besoin)
