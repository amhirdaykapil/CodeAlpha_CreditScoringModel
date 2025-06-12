import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("cs-training.csv")
df = df.apply(pd.to_numeric, errors='coerce')
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)
df.fillna(df.median(), inplace=True)

X = df.drop("SeriousDlqin2yrs", axis=1)
y = df["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print(" ROC AUC Score:", roc_auc_score(y_test, y_prob))

# random individual picking from dataset
random_index = random.randint(0, len(df) - 1)
individual_row = X.iloc[random_index:random_index + 1]
actual_id = int(df.iloc[random_index]["Id"])

print(f"\n{'-'*75}")
print(f"| Predicting for User ID: {actual_id:<57}|")
print(f"{'-'*75}")
for col in individual_row.columns:
    val = individual_row.iloc[0][col]
    print(f"| {col:<40} | {str(val):>25} |")
print(f"{'-'*75}")

ind_pred = model.predict(individual_row)[0]
ind_prob = model.predict_proba(individual_row)[0][1]

status = "SAFE" if ind_pred == 0 else "DEFAULT RISK"
print(f"| Prediction Result{'':<26} | {status} ({ind_prob:.2f} risk score) |")
print(f"{'-'*75}")
