import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
# Remplace le chemin par celui de ton fichier CSV
df = pd.read_csv('Rush 3\Cleaned_data\Clean_Credit_Data_Fichier_Clients.csv')
# Variables quantitatives
quant_vars = ['credit_amount', 'credit_term', 'age', 'income']
# Variables catégorielles
cat_vars = ['month', 'sex', 'education', 'product_type', 'region', 'family_status', 'phone_operator', 'having_children_flg', 'is_client']

# 1. Aperçu général
def eda_overview(df):
    print('Aperçu des données:')
    print(df.head())
    print('\nInfos:')
    print(df.info())
    print('\nStatistiques descriptives:')
    print(df.describe())
    print('\nValeurs manquantes:')
    print(df.isnull().sum())

# 2. Analyse univariée
def eda_univariate_quant(df):
    for var in quant_vars:
        plt.figure(figsize=(8,4))
        sns.histplot(df[var].dropna(), kde=True)
        plt.title(f'Distribution de {var}')
        plt.show()

def eda_univariate_cat(df):
    for var in cat_vars:
        plt.figure(figsize=(8,4))
        sns.countplot(x=var, data=df)
        plt.title(f'Repartition de {var}')
        plt.show()
        if var == 'product_type':
            top5 = df['product_type'].value_counts().head(5)
            print('Top 5 des product_type :')
            print(top5)

# 3. Matrice de corrélation
# Pour les variables numériques
def eda_corr(df):
    plt.figure(figsize=(10,8))
    corr = df[quant_vars].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Matrice de corrélation')
    plt.show()

# Utilisation :
eda_overview(df)
eda_univariate_quant(df)
eda_univariate_cat(df)
eda_corr(df)
