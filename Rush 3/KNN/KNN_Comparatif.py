import matplotlib.pyplot as plt
import pandas as pd

# Scores KNN Clean_Credit_Data_Fichier_Clients.csv
precision_1 = 0.08
recall_1 = 0.03
f1_1 = 0.04
accuracy_1 = 0.86

# Scores KNN Risque_Credit_Data_Fichier_Clients.csv
precision_2 = 0.55
recall_2 = 0.31
f1_2 = 0.40
accuracy_2 = 0.76

# Tableau comparatif
df_compare = pd.DataFrame({
    'Modèle': ['KNN_Clean_Credit', 'KNN_Risque_Credit'],
    'Precision mauvais client': [precision_1, precision_2],
    'Recall mauvais client': [recall_1, recall_2],
    'F1 mauvais client': [f1_1, f1_2],
    'Accuracy globale': [accuracy_1, accuracy_2]
})
print(df_compare)

# Barplot précision et rappel sur les mauvais clients
labels = ['KNN_Clean_Credit', 'KNN_Risque_Credit']
precision = [precision_1, precision_2]
recall = [recall_1, recall_2]

x = range(len(labels))
plt.figure(figsize=(8,5))
plt.bar(x, precision, width=0.4, label='Precision', align='center')
plt.bar([i+0.4 for i in x], recall, width=0.4, label='Recall', align='center')
plt.xticks([i+0.2 for i in x], labels)
plt.ylim(0,1)
plt.ylabel('Score')
plt.title('Comparaison Precision/Recall sur mauvais clients')
plt.legend()
plt.tight_layout()
plt.show()