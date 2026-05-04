import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import joblib
import os

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("circuit_exec_ms", axis=1)
y = df["circuit_exec_ms"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("quantumbench-circuit-exec-ms")

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0)
}

results = []

best_mae = float("inf")
best_model_name = None
best_model = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.set_tag("domain", "quantum_computing")

        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

        if mae < best_mae:
            best_mae = mae
            best_model_name = name
            best_model = model

# Save best model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

# Save JSON
output = {
    "experiment_name": "quantumbench-circuit-exec-ms",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "mae",
    "best_metric_value": best_mae
}

os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed")