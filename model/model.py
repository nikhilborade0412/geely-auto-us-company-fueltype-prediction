import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create pkl folder
os.makedirs("pkl", exist_ok=True)

# Load dataset
df = pd.read_csv("dataset/CarPrice_Assignment.csv")

# Drop unnecessary columns
df = df.drop(['car_ID', 'CarName'], axis=1)

# Convert target variable
df['fueltype'] = df['fueltype'].map({'gas': 0, 'diesel': 1})

# Select leakage-free strategic features
selected_features = [
    'enginesize',
    'compressionratio',
    'horsepower',
    'citympg',
    'highwaympg',
    'price',
    'curbweight',
    'stroke',
    'carlength',
    'carwidth'
]

X = df[selected_features]
y = df['fueltype']

# Save feature column order
feature_columns = X.columns

# Train-test split (stratified for imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression with class balancing
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save artifacts
pickle.dump(model, open("pkl/fueltype_model.pkl", "wb"))
pickle.dump(scaler, open("pkl/scaler.pkl", "wb"))
pickle.dump(feature_columns, open("pkl/feature_columns.pkl", "wb"))

print("\nLeakage-free model saved successfully.")