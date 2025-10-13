import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#######################
# Charger les données #
#######################

# Remplace le chemin par celui de ton fichier CSV
df = pd.read_csv('Rush 3\Cleaned_data\Clean_Credit_Data_Fichier_Clients.csv')

# Variables quantitatives
quant_vars = ['credit_amount', 'credit_term', 'age', 'income']

# Variables catégorielles
cat_vars = ['month', 'sex', 'education', 'product_type', 'region', 'family_status', 'phone_operator', 'having_children_flg', 'is_client']

# Nouvelles variables
df['debt_ratio'] = df['credit_amount'] / df['income']
df['credit_per_year_of_age'] = df['credit_amount'] / df['age']
df['term_to_age_ratio'] = df['credit_term'] / df['age']
df['children_income_ratio'] = df['having_children_flg'] * df['credit_amount'] / df['income']

# Variables risques
df["risk_score"] = 0

# Endettement
df.loc[df["credit_amount"] / df["income"] > 0.4, "risk_score"] += 3
df.loc[df["credit_amount"] / df["income"] > 0.6, "risk_score"] += 2 

# Durée du crédit
df.loc[df["credit_term"] > 60, "risk_score"] += 2
df.loc[df["credit_term"] > 120, "risk_score"] += 1

# Âge
df.loc[df["age"] < 25, "risk_score"] += 2
df.loc[df["age"] > 60, "risk_score"] += 2

# Enfants + revenus faibles
seuil_bas = df["income"].quantile(0.25)
df.loc[(df["having_children_flg"] == 1) & (df["income"] < seuil_bas), "risk_score"] += 2

# Revenus faibles
df.loc[df["income"] < seuil_bas, "risk_score"] += 3

# Éducation
low_edu = ["Primary", "Lower secondary"]
df.loc[df["education"].isin(low_edu), "risk_score"] += 2

# Produit risqué
risky_prod = ["Consumer credit", "Revolving"]
df.loc[df["product_type"].isin(risky_prod), "risk_score"] += 2

# Fidélité
df.loc[df["is_client"] == 0, "risk_score"] += 2

######################################################
# fonctions d'analyse exploratoire des données (EDA) #
######################################################

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

# 2.1 Analyse univariée quantitative
def eda_univariate_quant(df):
    for var in quant_vars:
        plt.figure(figsize=(8,4))
        sns.histplot(df[var].dropna(), kde=True)
        plt.title(f'Distribution de {var}')
        plt.show()

# 2.2 Analyse univariée catégorielle
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
def eda_corr(df):
    plt.figure(figsize=(10,8))
    corr = df[quant_vars].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Matrice de corrélation')
    plt.show()

# 4. Scatterplot : credit_amount vs income (avec seuil de debt_ratio)
def eda_scatter_credit_income(df):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='income', y='credit_amount', data=df, hue=(df['debt_ratio'] > 0.4))
    plt.axline((0, 0), slope=0.4, color='red', linestyle='--', label='debt_ratio=0.4')
    plt.title('Montant du crédit vs Revenu (debt_ratio>0.4 en couleur)')
    plt.legend()
    plt.show()

# 5. Boxplot : credit_term vs age
def eda_boxplot_term_age(df):
    plt.figure(figsize=(8,6))
    sns.boxplot(x=pd.cut(df['age'], bins=[18,30,40,50,60,80]), y='credit_term', data=df)
    plt.title('Durée du crédit selon tranches d\'âge')
    plt.xlabel('Tranche d\'âge')
    plt.ylabel('Durée du crédit')
    plt.show()

# 6. Boxplot : income par region
def eda_boxplot_income_region(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(x='region', y='income', data=df)
    plt.title('Distribution des revenus par région')
    plt.xticks(rotation=45)
    plt.show()

# 7. Boxplot : credit_amount par education
def eda_boxplot_credit_education(df):
    plt.figure(figsize=(8,6))
    sns.boxplot(x='education', y='credit_amount', data=df)
    plt.title('Montant du crédit selon le niveau d\'éducation')
    plt.show()

# 8. Barplot : product_type par sexe
def eda_barplot_product_type_sexe(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='product_type', hue='sex', data=df)
    plt.title('Répartition des produits par sexe')
    plt.xticks(rotation=45)
    plt.show()

##############################
# Code d'appel des fonctions #
##############################

# eda_overview(df)
# eda_univariate_quant(df)
# eda_univariate_cat(df)
# eda_corr(df)
# eda_scatter_credit_income(df)
# eda_boxplot_term_age(df)            # utiles ?
# eda_boxplot_income_region(df)       # utiles ?
# eda_boxplot_credit_education(df)    # utiles ?
# eda_barplot_product_type_sexe(df)

##################################
# Créer le dossier et le fichier #
##################################

# Créer le dossier Risque_data s'il n'existe pas
os.makedirs('Rush 3/Risque_data', exist_ok=True)

# Sauvegarder le fichier avec les risques
df.to_csv('Rush 3/Risque_data/Risque_Credit_Data_Fichier_Clients.csv', index=False)