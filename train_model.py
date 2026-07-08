"""
===========================================================
Maternal Health Risk Prediction System
===========================================================

Algorithm:
Random Forest Classifier

This module performs:

1. Load Dataset
2. Data Preprocessing
3. Train/Test Split
4. Random Forest Training
5. Model Evaluation
6. Save Trained Model

Figures (generated in Part 2):

• Accuracy Chart
• Confusion Matrix
• Classification Report
• Feature Importance
• ROC Curve
• Precision-Recall-F1 Chart
• Correlation Heatmap
• Dataset Distribution

===========================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import learning_curve

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# ===========================================================
# Create Required Folders
# ===========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ===========================================================
# Load Dataset
# ===========================================================

print("=" * 60)
print("Loading Maternal Health Dataset...")
print("=" * 60)

df = pd.read_csv("dataset/maternal_health.csv")

# Remove unwanted spaces in column names

df.columns = df.columns.str.strip()

print("\nDataset Loaded Successfully.")

print("\nFirst Five Records\n")

print(df.head())

print("\nDataset Shape")

print(df.shape)

print("\nDataset Information\n")

print(df.info())

print("\nMissing Values")

print(df.isnull().sum())

print("\nColumns")

print(df.columns.tolist())

# ===========================================================
# Feature Selection
# ===========================================================

print("\nSelecting Features...")

X = df.drop("RiskLevel", axis=1)

y = df["RiskLevel"]

print("\nInput Features")

print(X.columns.tolist())

print("\nTarget")

print("RiskLevel")

# ===========================================================
# Label Encoding
# ===========================================================

print("\nEncoding Labels...")

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nEncoded Classes")

for i, label in enumerate(encoder.classes_):
    print(f"{i} -> {label}")

# ===========================================================
# Train Test Split
# ===========================================================

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded

)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ===========================================================
# Build Random Forest
# ===========================================================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=10,

    random_state=42

)

model.fit(

    X_train,

    y_train

)

print("Training Completed Successfully.")

# ===========================================================
# Prediction
# ===========================================================

print("\nMaking Predictions...")

prediction = model.predict(X_test)

probability = model.predict_proba(X_test)

# ===========================================================
# Accuracy
# ===========================================================

accuracy = accuracy_score(

    y_test,

    prediction

)

print("\n")

print("=" * 60)

print(f"Model Accuracy : {accuracy*100:.2f}%")

print("=" * 60)

# ===========================================================
# Classification Report
# ===========================================================

report = classification_report(

    y_test,

    prediction,

    target_names=encoder.classes_,

    output_dict=True

)

print("\nClassification Report\n")

print(

classification_report(

    y_test,

    prediction,

    target_names=encoder.classes_

)

)

# ===========================================================
# Confusion Matrix
# ===========================================================

cm = confusion_matrix(

    y_test,

    prediction

)

print("\nConfusion Matrix\n")

print(cm)

# ===========================================================
# Save Model
# ===========================================================

print("\nSaving Model...")

joblib.dump(

    model,

    "models/risk_model.pkl"

)

joblib.dump(

    encoder,

    "models/encoder.pkl"

)

print("Model Saved Successfully.")

print("\nSaved Files")

print("models/risk_model.pkl")

print("models/encoder.pkl")
# ===========================================================
# VISUALIZATION SECTION
# ===========================================================

print("\nGenerating Evaluation Figures...")

# ===========================================================
# 1. Accuracy Chart
# ===========================================================

plt.figure(figsize=(6,5))

bars = plt.bar(
    ["Random Forest"],
    [accuracy*100],
    color="#2563eb"
)

plt.ylim(0,100)

plt.ylabel("Accuracy (%)")

plt.title("Random Forest Model Accuracy")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x()+bar.get_width()/2,
        height+1,
        f"{height:.2f}%",
        ha="center",
        fontsize=11
    )

plt.grid(axis="y", alpha=0.3)

plt.savefig(
    "results/accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✔ accuracy.png created")

# ===========================================================
# 2. Confusion Matrix
# ===========================================================

fig, ax = plt.subplots(figsize=(6,6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=False
)

plt.title("Confusion Matrix")

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✔ confusion_matrix.png created")

# ===========================================================
# 3. Classification Report Figure
# ===========================================================

report_df = pd.DataFrame(report).transpose()

fig, ax = plt.subplots(figsize=(10,5))

ax.axis("off")

table = ax.table(

    cellText=report_df.round(2).values,

    rowLabels=report_df.index,

    colLabels=report_df.columns,

    loc="center"

)

table.auto_set_font_size(False)

table.set_fontsize(10)

table.scale(1.2,2)

plt.title("Classification Report")

plt.savefig(

    "results/classification_report.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("✔ classification_report.png created")

# ===========================================================
# 4. Feature Importance
# ===========================================================

importance = pd.DataFrame({

    "Feature":X.columns,

    "Importance":model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

plt.figure(figsize=(8,5))

plt.barh(

    importance["Feature"],

    importance["Importance"],

    color="#22c55e"

)

plt.gca().invert_yaxis()

plt.xlabel("Importance Score")

plt.title("Feature Importance")

plt.grid(alpha=0.3)

plt.savefig(

    "results/feature_importance.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("✔ feature_importance.png created")

# ===========================================================
# 5. ROC Curve
# ===========================================================

y_test_bin = label_binarize(

    y_test,

    classes=[0,1,2]

)

colors = [

    "red",

    "orange",

    "green"

]

plt.figure(figsize=(7,6))

for i,color in enumerate(colors):

    fpr,tpr,_ = roc_curve(

        y_test_bin[:,i],

        probability[:,i]

    )

    roc_auc = auc(fpr,tpr)

    plt.plot(

        fpr,

        tpr,

        color=color,

        linewidth=2,

        label=f"{encoder.classes_[i]} (AUC={roc_auc:.2f})"

    )

plt.plot(

    [0,1],

    [0,1],

    linestyle="--",

    color="black"

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(alpha=0.3)

plt.savefig(

    "results/roc_curve.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("✔ roc_curve.png created")
# =====================================================
# Learning Curve
# =====================================================

train_sizes, train_scores, test_scores = learning_curve(
    model,
    X,
    y_encoded,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1,1.0,5),
    random_state=42
)

train_mean = train_scores.mean(axis=1)

test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8,5))

plt.plot(
    train_sizes,
    train_mean,
    marker='o',
    linewidth=2,
    label="Training Accuracy"
)

plt.plot(
    train_sizes,
    test_mean,
    marker='s',
    linewidth=2,
    label="Validation Accuracy"
)

plt.xlabel("Training Samples")

plt.ylabel("Accuracy")

plt.title("Learning Curve")

plt.legend()

plt.grid(True)

plt.savefig(
    "results/learning_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("✔ learning_curve.png created")

# ===========================================================
# 6. Precision Recall F1 Chart
# ===========================================================

metrics = report_df.iloc[:3][

    [

        "precision",

        "recall",

        "f1-score"

    ]

]

metrics.plot(

    kind="bar",

    figsize=(8,5)

)

plt.ylim(0,1)

plt.ylabel("Score")

plt.title("Precision Recall F1-score")

plt.grid(axis="y", alpha=0.3)

plt.savefig(

    "results/precision_recall_f1.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("✔ precision_recall_f1.png created")

# ===========================================================
# 7. Dataset Distribution
# ===========================================================

plt.figure(figsize=(6,5))

df["RiskLevel"].value_counts().plot(

    kind="bar",

    color=["#22c55e","#f59e0b","#ef4444"]

)

plt.ylabel("Samples")

plt.title("Risk Level Distribution")

plt.grid(axis="y", alpha=0.3)

plt.savefig(

    "results/class_distribution.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("✔ class_distribution.png created")

# ===========================================================
# 8. Correlation Heatmap
# ===========================================================

plt.figure(figsize=(8,6))

sns.heatmap(

    df.corr(numeric_only=True),

    annot=True,

    cmap="coolwarm",

    linewidths=0.5

)

plt.title("Feature Correlation Heatmap")

plt.savefig(

    "results/correlation_heatmap.png",

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print("✔ correlation_heatmap.png created")

# ===========================================================
# FINISHED
# ===========================================================

print("\n" + "="*60)
print("PROJECT TRAINING COMPLETED SUCCESSFULLY")
print("="*60)

print("\nGenerated Files:")

print("""
results/
│
├── accuracy.png
├── confusion_matrix.png
├── classification_report.png
├── feature_importance.png
├── roc_curve.png
├── precision_recall_f1.png
├── class_distribution.png
├── correlation_heatmap.png
└── learning_curve.png
           
""")
