"""
Maternal Health Risk Prediction Model

Algorithm:
Random Forest Classifier
"""

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


print("=" * 60)
print("Loading Dataset...")
print("=" * 60)


df = pd.read_csv("dataset/maternal_health.csv")

print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumns")

print(df.columns.tolist())



# ----------------------------
# Features
# ----------------------------

X = df.drop("RiskLevel", axis=1)

# ----------------------------
# Labels
# ----------------------------

y = df["RiskLevel"]



encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)



print("\nRisk Classes")

print(encoder.classes_)



# ----------------------------
# Split Dataset
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)



# ----------------------------
# Build Model
# ----------------------------

print("\nTraining Model...\n")

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=10,

    random_state=42

)

model.fit(X_train, y_train)



# ----------------------------
# Prediction
# ----------------------------

prediction = model.predict(X_test)



accuracy = accuracy_score(
    y_test,
    prediction
)

print("\nAccuracy")

print(round(accuracy * 100, 2), "%")



print("\nClassification Report\n")

print(classification_report(
    y_test,
    prediction
))



print("\nConfusion Matrix\n")

print(confusion_matrix(
    y_test,
    prediction
))



# ----------------------------
# Save Model
# ----------------------------

joblib.dump(
    model,
    "models/risk_model.pkl"
)

joblib.dump(
    encoder,
    "models/encoder.pkl"
)


print("\nModel Saved Successfully.")