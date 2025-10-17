import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# Charger vos données
df = pd.read_csv('Rush 4/Cleaned_data/Clean_Camp_Market.csv')

# Supprimer les colonnes non numériques ou non pertinentes pour le KNN
df = df.drop(['ID', 'Dt_Customer'], axis=1)

# Encoder les variables catégorielles
df = pd.get_dummies(df, columns=['Education', 'Marital_Status'])

# Supposez que la dernière colonne est la cible
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Séparer en jeu d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Courbe de précision pour différents k
accuracies = []
k_range = range(1, 21)
for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(X_train, y_train)
	y_pred_k = knn.predict(X_test)
	acc = accuracy_score(y_test, y_pred_k)
	accuracies.append(acc)

plt.figure(figsize=(8, 4))
plt.plot(k_range, accuracies, marker='o')
plt.title('Précision du KNN en fonction de k')
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Précision')
plt.grid(True)
plt.show()

# 2. Modèle final avec le meilleur k
best_k = np.argmax(accuracies) + 1
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 3. Afficher la précision
print(f"Meilleur k: {best_k}")
print("Accuracy:", accuracy_score(y_test, y_pred))

# 4. Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrice de confusion')
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.show()

# 5. Rapport de classification
print("\nRapport de classification:\n", classification_report(y_test, y_pred))

# 6. Visualisation PCA 2D (si possible)
try:
	pca = PCA(n_components=2)
	X_vis = pca.fit_transform(X_test)
	plt.figure(figsize=(8, 6))
	scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_pred, cmap='Set1', alpha=0.7)
	plt.title('Projection PCA des prédictions KNN')
	plt.xlabel('Composante principale 1')
	plt.ylabel('Composante principale 2')
	plt.legend(*scatter.legend_elements(), title="Classe prédite")
	plt.show()
except Exception as e:
	print("PCA non affichée :", e)

# 7. Exporter les prédictions dans un CSV
results = pd.DataFrame(X_test, columns=df.columns[:-1])
results['y_true'] = y_test
results['y_pred'] = y_pred
results.to_csv('Rush 4/Cleaned_data/KNN_predictions.csv', index=False)
print("\nPrédictions exportées dans Rush 4/Cleaned_data/KNN_predictions.csv")