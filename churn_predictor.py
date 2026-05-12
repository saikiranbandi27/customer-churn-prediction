# CUSTOMER CHURN PREDICTION

# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# LOAD DATASET

print("1. Loading Data...")

df = pd.read_csv("Telco-Customer-Churn.csv")

# DATA PREPROCESSING

print("\n2. Data Preprocessing...")

# Convert target column
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Drop customerID
df = df.drop("customerID", axis=1)

# Fix TotalCharges column
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# EDA

print("\n3. Exploratory Data Analysis...")

print(df["Churn"].value_counts(normalize=True))

# Churn distribution
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

# Gender vs churn
sns.countplot(x="gender", hue="Churn", data=df)
plt.title("Gender vs Churn")
plt.show()

# Tenure distribution
sns.histplot(df["tenure"], bins=50, kde=True)
plt.title("Tenure Distribution")
plt.show()

# FEATURE ENGINEERING

print("\n4. Feature Engineering...")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# FEATURE SCALING

num_cols = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(
    X_train[num_cols]
)

X_test[num_cols] = scaler.transform(
    X_test[num_cols]
)

# LOGISTIC REGRESSION

print("\n--- Logistic Regression ---")

lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

y_prob_lr = lr.predict_proba(X_test)[:, 1]

print(f"Accuracy  : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob_lr):.4f}")

# DECISION TREE

print("\n--- Decision Tree ---")

dt = DecisionTreeClassifier(
    max_depth=6,
    class_weight='balanced',
    random_state=42
)

dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

y_prob_dt = dt.predict_proba(X_test)[:, 1]

print(f"Accuracy  : {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred_dt):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob_dt):.4f}")

# RANDOM FOREST

print("\n--- Random Forest ---")

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=8,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced_subsample',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# Train model
rf.fit(X_train, y_train)

# Predict probabilities
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Threshold tuning
threshold = 0.40

# Convert probabilities to predictions
y_pred_rf = (y_prob_rf >= threshold).astype(int)

# =========================
# EVALUATION
# =========================

print(f"Accuracy  : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob_rf):.4f}")

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred_rf
))

print("\nConfusion Matrix:\n")

cm = confusion_matrix(
    y_test,
    y_pred_rf
)

print(cm)

# Heatmap
sns.heatmap(cm, annot=True, fmt='d')

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# FEATURE IMPORTANCE

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Features:\n")

print(importance.head(10))

# Plot feature importance
plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")

plt.show()

# SAVE MODEL

print("\nSaving Model...")

# Create folder
os.makedirs("models", exist_ok=True)

# Save model and preprocessing objects
joblib.dump(
    rf,
    "models/churn_random_forest.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/model_columns.pkl"
)

joblib.dump(
    threshold,
    "models/threshold.pkl"
)

print("Model saved successfully!")

# PREDICTION FUNCTION

def predict_churn(new_data):

    # Load saved files
    model = joblib.load(
        "models/churn_random_forest.pkl"
    )

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    model_columns = joblib.load(
        "models/model_columns.pkl"
    )

    threshold = joblib.load(
        "models/threshold.pkl"
    )

    # Copy dataframe
    df_new = new_data.copy()

    # Fix TotalCharges
    if "TotalCharges" in df_new.columns:

        df_new["TotalCharges"] = pd.to_numeric(
            df_new["TotalCharges"],
            errors="coerce"
        )

        df_new["TotalCharges"] = df_new[
            "TotalCharges"
        ].fillna(
            df_new["TotalCharges"].median()
        )

    # One-hot encode
    df_new = pd.get_dummies(
        df_new,
        drop_first=True
    )

    # Match training columns
    df_new = df_new.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Scale numerical columns
    num_cols = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    df_new[num_cols] = scaler.transform(
        df_new[num_cols]
    )

    # Predict probability
    probabilities = model.predict_proba(
        df_new
    )[:,1]

    # Convert to prediction
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions, probabilities

print("\nProject Completed Successfully!")
