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
# Normalisation du risk_score entre 0 et 1
min_score = df["risk_score"].min()
max_score = df["risk_score"].max()
if max_score > min_score:
    df["risk_score"] = (df["risk_score"] - min_score) / (max_score - min_score)
# Ajout d'une colonne bad_client_risk_score (True/False) selon risk_score > 0.5
df['bad_client_risk_score'] = df['risk_score'].apply(lambda x: 'True' if x > 0.5 else 'False')

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

###########################################################################
# Analyse de la proportion de mauvais clients selon bad_client_risk_score #
###########################################################################

# 9. Barplot : month par bad_client_risk_score
def eda_barplot_month_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='month', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des month par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 10. Barplot : credit_amount par bad_client_risk_score
def eda_barplot_credit_amount_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='credit_amount', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des credit_amount par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 11. Barplot : credit_term par bad_client_risk_score
def eda_barplot_credit_term_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='credit_term', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des credit_term par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 12. Barplot : age par bad_client_risk_score
def eda_barplot_age_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='age', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des age par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 13. Barplot : sex par bad_client_risk_score
def eda_barplot_sex_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='sex', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des sex par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 14. Barplot : education par bad_client_risk_score
def eda_barplot_education_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='education', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des education par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 15. Barplot : product_type par bad_client_risk_score
def eda_barplot_product_type_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='product_type', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des produits par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 16. Barplot : having_children_flg par bad_client_risk_score
def eda_barplot_having_children_flg_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='having_children_flg', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des having_children_flg par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 17. Barplot : region par bad_client_risk_score
def eda_barplot_region_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='region', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des region par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 18. Barplot : region par bad_client_risk_score
def eda_barplot_region_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='region', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des region par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 19. Barplot : income par bad_client_risk_score
def eda_barplot_income_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='income', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des income par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 19. Barplot : family_status par bad_client_risk_score
def eda_barplot_family_status_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='family_status', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des family_status par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 20. Barplot : phone_operator par bad_client_risk_score
def eda_barplot_phone_operator_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='phone_operator', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des phone_operator par bad_client_risk_score')
    plt.xticks(rotation=45)
    plt.show()

# 21. Barplot : is_client par bad_client_risk_score
def eda_barplot_is_client_bad_client_risk_score(df):
    plt.figure(figsize=(8,6))
    sns.countplot(x='is_client', hue='bad_client_risk_score', data=df)
    plt.title('Répartition des is_client par bad_client_risk_score')
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
# eda_barplot_month_bad_client_risk_score(df)
# eda_barplot_credit_amount_bad_client_risk_score(df)
# eda_barplot_credit_term_bad_client_risk_score(df)
# eda_barplot_age_bad_client_risk_score(df)
# eda_barplot_sex_bad_client_risk_score(df)
# eda_barplot_education_bad_client_risk_score(df)
# eda_barplot_product_type_bad_client_risk_score(df)
# eda_barplot_having_children_flg_bad_client_risk_score(df)
# eda_barplot_region_bad_client_risk_score(df)
# eda_barplot_region_bad_client_risk_score(df)
# eda_barplot_income_bad_client_risk_score(df)
# eda_barplot_family_status_bad_client_risk_score(df)
# eda_barplot_phone_operator_bad_client_risk_score(df)
# eda_barplot_is_client_bad_client_risk_score(df)


##################################
# Créer le dossier et le fichier #
##################################

# Créer le dossier Risque_data s'il n'existe pas
os.makedirs('Rush 3/Risque_data', exist_ok=True)

# Sauvegarder le fichier avec les risques
df.to_csv('Rush 3/Risque_data/Risque_Credit_Data_Fichier_Clients.csv', index=False)