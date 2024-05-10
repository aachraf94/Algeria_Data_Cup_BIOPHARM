import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

# Charger les données d'entraînement et de test
train_data = pd.read_csv('train_BIOPHARM.csv')
test_data = pd.read_csv('test_BIOPHARM.csv')
submission_format = pd.read_csv('Sample_submission_BIOPHARM.csv')

# Prétraitement des données
# Pour simplifier, supposons que le prétraitement inclut le retrait des colonnes qui ne sont pas nécessaires pour l'entraînement et le test
# De plus, l'encodage des variables catégorielles et la gestion des valeurs manquantes devraient être effectués dans un scénario réel
X_train = train_data.drop(['ID', 'Date', 'Target'], axis=1)
y_train = train_data['Target']
X_test = test_data.drop(['ID', 'Date'], axis=1)

# Diviser les données d'entraînement en ensembles d'entraînement et de validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialiser le classificateur Gradient Boosting
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Entraîner le modèle
gbc.fit(X_train_split, y_train_split)

# Faire des prédictions sur l'ensemble de validation
val_predictions = gbc.predict(X_val)
val_probabilities = gbc.predict_proba(X_val)

# Évaluer le modèle en utilisant l'exactitude et la perte logarithmique
accuracy = accuracy_score(y_val, val_predictions)
logloss = log_loss(y_val, val_probabilities)

print(f'Exactitude sur l\'ensemble de validation: {accuracy}')
print(f'Perte logarithmique sur l\'ensemble de validation: {logloss}')

# Faire des prédictions sur l'ensemble de test
test_predictions = gbc.predict(X_test)
test_probabilities = gbc.predict_proba(X_test)

# Créer un DataFrame de soumission
submission_df = pd.DataFrame(test_probabilities, columns=gbc.classes_)
submission_df['ID'] = test_data['ID']

# Réorganiser les colonnes pour correspondre au format de soumission
submission_df = submission_df[['ID'] + list(gbc.classes_)]

# Enregistrer le DataFrame de soumission au format CSV
submission_df.to_csv('submission_BIOPHARM.csv', index=False)
