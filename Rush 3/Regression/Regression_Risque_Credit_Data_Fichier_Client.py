import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# Charger les données
df = pd.read_csv('Rush 3\Risque_data\Risque_Credit_Data_Fichier_Clients.csv')

cat_features = ['sex', 'education', 'product_type', 'region', 'family_status', 'phone_operator']
for col in cat_features:
	df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Séparer les features et la cible
X = df.drop('bad_client_risk_score', axis=1)
y = df['bad_client_risk_score']

# Séparer en train et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle de régression logistique
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prédire sur le jeu de test
y_pred = model.predict(X_test)

# Afficher le rapport de classification
print(classification_report(y_test, y_pred))