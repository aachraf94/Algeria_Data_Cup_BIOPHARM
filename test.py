import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load the data
train_data = pd.read_csv('cleaned_train_BIOPHARM.csv')
test_data = pd.read_csv('test_BIOPHARM.csv')
submission_data = pd.read_csv('Sample_submission_BIOPHARM.csv')

# Encode categorical variables
le = LabelEncoder()

# Convert categorical columns to codes (integer labels)
train_data['Lieu'] = train_data['Lieu'].astype('category').cat.codes
test_data['Lieu'] = test_data['Lieu'].astype('category').cat.codes

# Define features and target
features = ['Lieu', 'Valeur', 'Nouvelle valeur', 'Ancienne valeur']
X_train = train_data[features]
y_train = train_data['Event type']
X_test = test_data[features]

# Initialize and train the XGBoost classifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, y_train)

# Predict probabilities for each event type in the test data
probabilities = xgb_classifier.predict_proba(X_test)

# Update the submission file with predicted probabilities for each index
for i, index in enumerate(submission_data['Index']):
    submission_data.loc[i, submission_data.columns[1:]] = probabilities[i]

# Save the updated submission file
submission_data.to_csv('Updated_submission_BIOPHARM.csv', index=False)
