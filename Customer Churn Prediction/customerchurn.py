import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Load data
df = pd.read_csv("Churn_Modelling.csv")

# Drop columns
df.drop(["RowNumber","CustomerId","Surname"], axis=1, inplace=True)

# Encoding
le = LabelEncoder()
df["Geography"] = le.fit_transform(df["Geography"])
df["Gender"] = le.fit_transform(df["Gender"])

# Feature engineering
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
df["AgeTenureRatio"] = df["Age"] / (df["Tenure"] + 1)

# Split
X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Better RF model
model = RandomForestClassifier(
    n_estimators=250,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]

# Results
print("Accuracy:", accuracy_score(y_test, pred))
print("ROC AUC:", roc_auc_score(y_test, prob))

print(classification_report(y_test, pred))

# Confusion Matrix
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()