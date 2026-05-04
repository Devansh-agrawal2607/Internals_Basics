import mlflow
from mlflow.tracking import MlflowClient
import json
import os

client = MlflowClient()

model_name = "quantumbench-circuit-exec-ms-predictor"

# Get all versions
versions = client.search_model_versions(f"name='{model_name}'")

# Sort versions
versions = sorted(versions, key=lambda v: int(v.version))

# Assume:
champion_version = int(versions[0].version)
challenger_version = int(versions[-1].version)

# Get metrics
def get_mae(version):
    run_id = version.run_id
    run = mlflow.get_run(run_id)
    return run.data.metrics["mae"]

champion_mae = get_mae(versions[0])
challenger_mae = get_mae(versions[-1])

# Compare
if challenger_mae < champion_mae:
    client.set_registered_model_alias(model_name, "champion", challenger_version)
    action = "promoted"
    final_champion = challenger_version
else:
    client.set_registered_model_alias(model_name, "champion", champion_version)
    action = "kept"
    final_champion = champion_version

# Prepare JSON
output = {
    "registered_model_name": model_name,
    "alias_name": "champion",
    "champion_version": final_champion,
    "challenger_version": challenger_version,
    "action": action
}

# Save JSON
os.makedirs("results", exist_ok=True)

with open("results/step4_s7.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 4 completed successfully")