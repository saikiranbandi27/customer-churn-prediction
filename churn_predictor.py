# Basic imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

print("1. Loading Data...")
df = pd.read_csv("../Telco-Customer-Churn.csv")

print("\n2. Data Preprocessing...")
# Map target to binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop customerID
df = df.drop('customerID', axis=1)

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

print("\n3. Exploratory Data Analysis (EDA)...")
# (We will just print some basic stats so it runs cleanly without blocking the terminal with plots, 
# but the code for plots is included for the notebook/user to run)
print(df['Churn'].value_counts(normalize=True))

# Uncomment below to see plots when running locally:
# sns.countplot(x='Churn', data=df)
# plt.show()
# sns.countplot(x='Churn', hue='gender', data=df)
# plt.show()
# sns.histplot(df['tenure'], bins=50)
# plt.show()

print("\n4. Feature Engineering...")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Dummy encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize numerical features
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

print("\n5. Model Training & Evaluation...")

# We use class_weight='balanced' to handle the imbalanced dataset (which gives better recall/F1!)

# --- 1. Logistic Regression ---
print("\n--- Logistic Regression ---")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

print(f"Accuracy  : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob_lr):.4f}")


# --- 2. Decision Tree ---
print("\n--- Decision Tree ---")
# Tuning parameters slightly to avoid overfitting and improve performance
dt = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

print(f"Accuracy  : {accuracy_score(y_test, y_pred_dt):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_dt):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_dt):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred_dt):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob_dt):.4f}")


# --- 3. Random Forest ---
print("\n--- Random Forest ---")
# Using improved hyperparameters and class_weight to get better results
rf = RandomForestClassifier(
    n_estimators=300, 
    max_depth=10, 
    min_samples_split=10, 
    min_samples_leaf=4, 
    class_weight='balanced', 
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print(f"Accuracy  : {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision : {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_prob_rf):.4f}")

print("\nOverall, the tuned Random Forest with balanced class weights provides the best balance of Precision and Recall!")
