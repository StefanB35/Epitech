import pandas as pd

# Charger les fichiers CSV
credit_data = pd.read_csv('Rush 3\Cleaned_data\Clean_Credit_Data_Fichier_Clients.csv')
risque_data = pd.read_csv('Rush 3\Risque_data\Risque_Credit_Data_Fichier_Clients.csv')

# Correction : selon la valeur réelle des colonnes (adapter si besoin)
# Pour bad_client_target : 1, '1', ou 'Oui'
if 'bad_client_target' in credit_data.columns:
	mauvais_credit = credit_data[(credit_data['bad_client_target'] == 1) | (credit_data['bad_client_target'] == '1') | (credit_data['bad_client_target'] == 'Oui')]
else:
	mauvais_credit = pd.DataFrame()

# Pour bad_client_risk_score : 'Oui', 1, 'True', 'true'
if 'bad_client_risk_score' in risque_data.columns:
	mauvais_risque = risque_data[(risque_data['bad_client_risk_score'] == 'Oui') |
								 (risque_data['bad_client_risk_score'] == 1) |
								 (risque_data['bad_client_risk_score'] == True) |
								 (risque_data['bad_client_risk_score'] == 'True') |
								 (risque_data['bad_client_risk_score'] == 'true')]
else:
	mauvais_risque = pd.DataFrame()

# Comparer les mauvais clients par leur identifiant
ids_mauvais_credit = set(mauvais_credit['ID'])
ids_mauvais_risque = set(mauvais_risque['ID'])

# Clients mauvais dans les deux fichiers
mauvais_dans_les_deux = ids_mauvais_credit & ids_mauvais_risque

# Clients mauvais uniquement dans Credit_data_fichier
mauvais_uniquement_credit = ids_mauvais_credit - ids_mauvais_risque

# Clients mauvais uniquement dans Risque_credit_data_fichier
mauvais_uniquement_risque = ids_mauvais_risque - ids_mauvais_credit

print(f"Mauvais clients dans les deux fichiers ({len(mauvais_dans_les_deux)}) :", mauvais_dans_les_deux)
print(f"Mauvais clients uniquement dans Clean_Credit_Data_Fichier_Clients ({len(mauvais_uniquement_credit)}) :", mauvais_uniquement_credit)
print(f"Mauvais clients uniquement dans Risque_Credit_Data_Fichier_Clients ({len(mauvais_uniquement_risque)}) :", mauvais_uniquement_risque)