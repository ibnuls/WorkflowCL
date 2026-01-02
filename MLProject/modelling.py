import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("dataset_preprocessing.csv")

X = df.drop("Churn Label", axis=1)
y = df["Churn Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================
# MLflow AUTLOG
# =====================
mlflow.set_experiment("ChurnPredictionBasic")

with mlflow.start_run():
    mlflow.autolog()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
