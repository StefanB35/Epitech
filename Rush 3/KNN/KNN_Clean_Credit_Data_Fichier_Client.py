import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Charger les données
df = pd.read_csv('Rush 3/Cleaned_data/Clean_Credit_Data_Fichier_Clients.csv')

# Sélectionner les colonnes numériques et catégorielles pertinentes
features = ['credit_amount', 'credit_term', 'age', 'income', 'having_children_flg', 'is_client']
cat_features = ['sex', 'education', 'product_type', 'region', 'family_status', 'phone_operator']

# Encodage des variables catégorielles
for col in cat_features:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    features.append(col)

# Définir X et y
X = df[features]
y = df['bad_client_target']

# Séparer en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardiser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer et entraîner le modèle KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prédire sur le test set
y_pred = knn.predict(X_test)

# Afficher les résultats
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))