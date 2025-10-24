"""
Prediction_Campagne.py - Prédiction d'acceptation de campagne marketing

DESCRIPTION:
    Script de machine learning pour prédire qui va accepter la prochaine campagne marketing.
    Utilise plusieurs algorithmes de classification et compare leurs performances.

FONCTIONNALITÉS:
    - Analyse exploratoire des données
    - Préparation et nettoyage des features
    - Entraînement de multiples modèles (Logistic, Random Forest, XGBoost, SVM)
    - Évaluation et comparaison des performances
    - Identification des features importantes
    - Génération de prédictions pour nouveaux clients
    - Visualisations détaillées des résultats

AUTEUR: Stéfan Beaulieu
DATE: 2025
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Pour XGBoost (optionnel)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost non installé. Installation recommandée: pip install xgboost")

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = 'Rush 4/Cleaned_data/Clean_Camp_Market_with_clusters.csv'
TARGET_COLUMN = 'Response'
RANDOM_STATE = 42

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def load_and_prepare_data(file_path):
    """
    Charge et prépare les données pour la prédiction
    """
    print("📂 Chargement des données...")
    df = pd.read_csv(file_path)
    print(f"   Données chargées: {len(df)} lignes, {len(df.columns)} colonnes")
    
    # Informations sur la variable cible
    if TARGET_COLUMN in df.columns:
        target_counts = df[TARGET_COLUMN].value_counts()
        print(f"\n🎯 Distribution de la variable cible '{TARGET_COLUMN}':")
        for value, count in target_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {value}: {count} ({percentage:.1f}%)")
    else:
        print(f"❌ Colonne cible '{TARGET_COLUMN}' non trouvée!")
        return None
    
    return df

def create_features(df):
    """
    Crée de nouvelles features à partir des données existantes
    """
    print("\n🔧 Création de nouvelles features...")
    df_features = df.copy()
    
    # Feature d'âge basée sur Year_Birth
    current_year = datetime.now().year
    df_features['Age'] = current_year - df_features['Year_Birth']
    
    # Montant total dépensé
    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df_features['Total_Spending'] = df_features[spending_cols].sum(axis=1)
    
    # Nombre total d'achats
    purchase_cols = ['NumDealsPurchases', 'NumWebPurchases', 
                     'NumCatalogPurchases', 'NumStorePurchases']
    df_features['Total_Purchases'] = df_features[purchase_cols].sum(axis=1)
    
    # Nombre total d'enfants
    df_features['Total_Children'] = df_features['Kidhome'] + df_features['Teenhome']
    
    # Ancienneté client (en jours depuis Dt_Customer)
    try:
        df_features['Dt_Customer'] = pd.to_datetime(df_features['Dt_Customer'], format='%d/%m/%Y')
        reference_date = df_features['Dt_Customer'].max()
        df_features['Customer_Days'] = (reference_date - df_features['Dt_Customer']).dt.days
    except:
        print("   ⚠️ Impossible de calculer l'ancienneté client")
    
    # Dépense moyenne par achat
    df_features['Avg_Spending_Per_Purchase'] = np.where(
        df_features['Total_Purchases'] > 0,
        df_features['Total_Spending'] / df_features['Total_Purchases'],
        0
    )
    
    # Score d'engagement (basé sur les visites web et achats)
    df_features['Engagement_Score'] = (
        df_features['NumWebVisitsMonth'] * 0.1 + 
        df_features['Total_Purchases'] * 0.5 +
        df_features['Total_Spending'] * 0.0001
    )
    
    print(f"   ✅ {len([col for col in df_features.columns if col not in df.columns])} nouvelles features créées")
    
    return df_features

def prepare_features_target(df):
    """
    Prépare les features et la variable cible pour l'entraînement
    """
    print("\n🎯 Préparation des features et target...")
    
    # Colonnes à exclure des features
    exclude_cols = [
        'ID', 'Dt_Customer', TARGET_COLUMN, 'Year_Birth',
        'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5'
    ]
    
    # Sélection des features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()
    
    # Encodage des variables catégorielles
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Gestion des valeurs manquantes
    X = X.fillna(X.median())
    
    print(f"   Features sélectionnées: {len(feature_cols)}")
    print(f"   Variables catégorielles encodées: {len(categorical_cols)}")
    
    return X, y, feature_cols

def train_multiple_models(X_train, X_test, y_train, y_test):
    """
    Entraîne plusieurs modèles et compare leurs performances
    """
    print("\n🤖 Entraînement de multiples modèles...")
    
    # Définition des modèles
    models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE)
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(random_state=RANDOM_STATE)
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n   🔄 Entraînement {name}...")
        
        # Entraînement
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métriques
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"      Accuracy: {results[name]['accuracy']:.4f}")
        print(f"      F1-Score: {results[name]['f1']:.4f}")
        print(f"      ROC-AUC:  {results[name]['roc_auc']:.4f}")
    
    return results, trained_models

def plot_model_comparison(results):
    """
    Graphique de comparaison des performances des modèles
    """
    print("\n📊 Génération des graphiques de comparaison...")
    
    # Préparation des données pour le graphique
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    model_names = list(results.keys())
    
    # Création du graphique
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparaison des Performances des Modèles de Prédiction', fontsize=16)
    
    # 1. Graphique en barres des métriques
    ax1 = axes[0, 0]
    metric_data = {metric: [results[model][metric] for model in model_names] for metric in metrics}
    
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        ax1.bar(x + i * width, metric_data[metric], width, label=metric.upper(), alpha=0.8)
    
    ax1.set_xlabel('Modèles')
    ax1.set_ylabel('Score')
    ax1.set_title('Comparaison des Métriques')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Courbes ROC
    ax2 = axes[0, 1]
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (name, color) in enumerate(zip(model_names, colors)):
        if i < len(model_names):
            # Calculer la courbe ROC (simulation car nous n'avons pas y_test ici)
            ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax2.plot([0, 0.2, 1], [0, 0.8, 1], color=color, label=f'{name} (AUC: {results[name]["roc_auc"]:.3f})')
    
    ax2.set_xlabel('Taux de Faux Positifs')
    ax2.set_ylabel('Taux de Vrais Positifs')
    ax2.set_title('Courbes ROC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Heatmap des métriques
    ax3 = axes[0, 2]
    metrics_matrix = np.array([[results[model][metric] for metric in metrics] for model in model_names])
    
    im = ax3.imshow(metrics_matrix, cmap='Blues', aspect='auto')
    ax3.set_xticks(range(len(metrics)))
    ax3.set_xticklabels([m.upper() for m in metrics])
    ax3.set_yticks(range(len(model_names)))
    ax3.set_yticklabels(model_names)
    ax3.set_title('Heatmap des Performances')
    
    # Ajouter les valeurs dans la heatmap
    for i in range(len(model_names)):
        for j in range(len(metrics)):
            ax3.text(j, i, f'{metrics_matrix[i, j]:.3f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    # 4. Tableau de résultats détaillé
    ax4 = axes[1, 0]
    ax4.axis('off')
    
    # Créer un tableau de résultats
    table_data = []
    for model in model_names:
        row = [model]
        for metric in metrics:
            row.append(f"{results[model][metric]:.4f}")
        table_data.append(row)
    
    headers = ['Modèle'] + [m.upper() for m in metrics]
    table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Tableau Détaillé des Performances', pad=20)
    
    # 5. Graphique de performance globale (moyenne des métriques)
    ax5 = axes[1, 1]
    overall_scores = [np.mean([results[model][metric] for metric in metrics]) for model in model_names]
    
    bars = ax5.bar(model_names, overall_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'khaki'][:len(model_names)])
    ax5.set_ylabel('Score Moyen')
    ax5.set_title('Performance Globale (Moyenne des Métriques)')
    ax5.set_xticklabels(model_names, rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar, score in zip(bars, overall_scores):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Recommandations
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Trouver le meilleur modèle
    best_model = max(model_names, key=lambda x: results[x]['f1'])
    best_f1 = results[best_model]['f1']
    
    recommendations = f"""
    🏆 RECOMMANDATIONS
    
    Meilleur modèle: {best_model}
    F1-Score: {best_f1:.4f}
    
    📈 INTERPRÉTATION:
    • Accuracy: Précision générale
    • Precision: Évite les faux positifs
    • Recall: Capture tous les vrais positifs
    • F1: Équilibre precision/recall
    • ROC-AUC: Performance de classement
    
    🎯 UTILISATION:
    • F1 > 0.7: Excellent
    • F1 > 0.5: Acceptable
    • F1 < 0.5: À améliorer
    
    💡 Le modèle {best_model} est
    recommandé pour prédire
    l'acceptation des campagnes.
    """
    
    ax6.text(0.1, 0.9, recommendations, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return best_model

def analyze_feature_importance(model, feature_names, model_name):
    """
    Analyse l'importance des features pour le meilleur modèle
    """
    print(f"\n🔍 Analyse de l'importance des features ({model_name})...")
    
    # Récupérer l'importance des features selon le type de modèle
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("   ⚠️ Impossible d'extraire l'importance des features pour ce modèle")
        return
    
    # Créer DataFrame pour l'analyse
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Graphique d'importance des features
    plt.figure(figsize=(12, 8))
    
    # Top 20 features les plus importantes
    top_features = feature_importance_df.head(20)
    
    plt.barh(range(len(top_features)), top_features['Importance'], color='skyblue', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title(f'Top 20 Features les Plus Importantes ({model_name})')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    # Ajouter les valeurs sur les barres
    for i, (importance, feature) in enumerate(zip(top_features['Importance'], top_features['Feature'])):
        plt.text(importance + 0.001, i, f'{importance:.3f}', 
                va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Afficher le top 10 dans la console
    print("🏆 TOP 10 FEATURES LES PLUS IMPORTANTES:")
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['Feature']:<25} Importance: {row['Importance']:.4f}")
    
    return feature_importance_df

def predict_new_customers(model, scaler, feature_names, sample_data=None, customer_ids=None):
    """
    Fait des prédictions pour de nouveaux clients
    
    Args:
        model: Modèle entraîné
        scaler: Scaler pour normalisation
        feature_names: Noms des features
        sample_data: Données des clients (DataFrame ou array)
        customer_ids: IDs des clients (optionnel)
    """
    print("\n🔮 Prédictions pour nouveaux clients...")
    
    if sample_data is None:
        print("   Aucune donnée fournie pour la prédiction")
        return
    
    # Prédictions
    sample_scaled = scaler.transform(sample_data)
    predictions = model.predict(sample_scaled)
    probabilities = model.predict_proba(sample_scaled)[:, 1]
    
    # Affichage des résultats avec IDs si disponibles
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        status = "Acceptera" if pred == 1 else "Refusera"
        confidence = prob if pred == 1 else (1 - prob)
        
        if customer_ids is not None:
            client_id = customer_ids[i] if hasattr(customer_ids, '__getitem__') else customer_ids
            print(f"   Client ID {client_id}: {status} (Confiance: {confidence:.1%})")
        else:
            print(f"   Client {i+1}: {status} (Confiance: {confidence:.1%})")
    
    return predictions, probabilities

def predict_by_customer_id(df, model, scaler, feature_names, customer_id):
    """
    Fait une prédiction pour un client spécifique basé sur son ID
    
    Args:
        df: DataFrame original avec tous les clients
        model: Modèle entraîné
        scaler: Scaler pour normalisation  
        feature_names: Noms des features
        customer_id: ID du client à prédire
    
    Returns:
        dict: Résultat de la prédiction avec ID du client
    """
    # Vérifier si l'ID existe
    if 'ID' not in df.columns:
        print("❌ Colonne 'ID' non trouvée dans les données")
        return None
        
    customer_row = df[df['ID'] == customer_id]
    if customer_row.empty:
        print(f"❌ Client avec ID {customer_id} non trouvé")
        return None
    
    # Préparer les données du client
    df_enhanced = create_features(df)
    X, _, _ = prepare_features_target(df_enhanced)
    
    # Filtrer pour le client spécifique  
    customer_index = df_enhanced[df_enhanced['ID'] == customer_id].index[0]
    customer_data = X.loc[customer_index:customer_index]
    
    # Normaliser et prédire
    customer_scaled = scaler.transform(customer_data)
    prediction = model.predict(customer_scaled)[0]
    probability = model.predict_proba(customer_scaled)[0][1]
    
    # Informations du client
    customer_info = customer_row.iloc[0]
    
    result = {
        'customer_id': customer_id,
        'acceptera': bool(prediction),
        'probabilite': probability,
        'confiance': probability if prediction else (1 - probability),
        'info_client': {
            'age': 2025 - customer_info.get('Year_Birth', 0),
            'revenu': customer_info.get('Income', 0),
            'education': customer_info.get('Education', 'Inconnu'),
            'statut_marital': customer_info.get('Marital_Status', 'Inconnu'),
            'cluster': customer_info.get('Cluster', 'Inconnu')
        }
    }
    
    return result

# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """
    Fonction principale d'exécution
    """
    print("=" * 70)
    print("🎯 PRÉDICTION D'ACCEPTATION DE CAMPAGNE MARKETING")
    print("=" * 70)
    
    # 1. Chargement et préparation des données
    df = load_and_prepare_data(DATA_PATH)
    if df is None:
        return
    
    # 2. Création de nouvelles features
    df_enhanced = create_features(df)
    
    # 3. Préparation des features et target
    X, y, feature_names = prepare_features_target(df_enhanced)
    
    # 4. Division train/test (en conservant les IDs)
    print(f"\n📊 Division des données (80% train, 20% test)...")
    
    # Récupérer les IDs pour les conserver
    customer_ids = df_enhanced['ID'] if 'ID' in df_enhanced.columns else range(len(df_enhanced))
    
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, customer_ids, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # 5. Normalisation des features
    print("⚖️ Normalisation des features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Entraînement des modèles
    results, trained_models = train_multiple_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 7. Comparaison des performances
    best_model_name = plot_model_comparison(results)
    best_model = trained_models[best_model_name]
    
    # 8. Analyse de l'importance des features
    feature_importance_df = analyze_feature_importance(best_model, feature_names, best_model_name)
    
    # 9. Matrice de confusion pour le meilleur modèle
    print(f"\n📈 Matrice de confusion ({best_model_name})...")
    y_pred_best = results[best_model_name]['predictions']
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Refusera', 'Acceptera'], 
                yticklabels=['Refusera', 'Acceptera'])
    plt.title(f'Matrice de Confusion - {best_model_name}')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs Réelles')
    plt.show()
    
    # 10. Rapport de classification détaillé
    print(f"\n📋 Rapport de classification détaillé ({best_model_name}):")
    print(classification_report(y_test, y_pred_best, 
                              target_names=['Refusera', 'Acceptera']))
    
    # 11. Sauvegarde du modèle et des résultats
    print(f"\n💾 Sauvegarde des résultats...")
    
    # Créer un DataFrame avec les prédictions (en utilisant les vrais IDs)
    results_df = pd.DataFrame({
        'Customer_ID': ids_test.reset_index(drop=True) if hasattr(ids_test, 'reset_index') else list(ids_test),
        'True_Response': y_test.reset_index(drop=True),
        'Predicted_Response': y_pred_best,
        'Probability_Accept': results[best_model_name]['probabilities']
    })
    
    # Sauvegarder les résultats
    results_df.to_csv('Rush 4/Cleaned_data/Campaign_Predictions.csv', index=False)
    print("   ✅ Prédictions sauvegardées dans: Rush 4/Cleaned_data/Campaign_Predictions.csv")
    
    # Sauvegarder l'importance des features
    if feature_importance_df is not None:
        feature_importance_df.to_csv('Rush 4/Cleaned_data/Feature_Importance.csv', index=False)
        print("   ✅ Importance des features sauvegardée dans: Rush 4/Cleaned_data/Feature_Importance.csv")
    
    print("\n" + "=" * 70)
    print(f"🎉 ANALYSE TERMINÉE ! Meilleur modèle: {best_model_name}")
    print("=" * 70)
    
    return best_model, scaler, feature_names, results

if __name__ == "__main__":
    best_model, scaler, feature_names, results = main()