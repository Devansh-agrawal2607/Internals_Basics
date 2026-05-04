import mlflow
from mlflow.tracking import MlflowClient
import json
import os

client = MlflowClient()

# Get experiment
experiment = mlflow.get_experiment_by_name("quantumbench-circuit-exec-ms")

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Get best run
best_run = runs.sort_values("metrics.mae").iloc[0]
run_id = best_run.run_id

# ✅ Get model name dynamically (IMPORTANT FIX)
model_name_logged = best_run["params.model"]

# Correct URI
model_uri = f"runs:/{run_id}/{model_name_logged}"

# Registered model name
registered_model_name = "quantumbench-circuit-exec-ms-predictor"

# Register model
result = mlflow.register_model(model_uri, registered_model_name)

# Prepare output JSON
output = {
    "registered_model_name": registered_model_name,
    "version": result.version,
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": float(best_run["metrics.mae"])
}

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Save JSON automatically
with open("results/step3_s6.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 3 completed successfully")