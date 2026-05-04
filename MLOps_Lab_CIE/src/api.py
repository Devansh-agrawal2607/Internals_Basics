from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
import os

app = FastAPI()

# Load model
model = joblib.load("models/best_model.pkl")

# Input validation
class InputData(BaseModel):
    qubit_count: int = Field(..., ge=5, le=200)
    gate_depth: int = Field(..., ge=10, le=1000)
    error_rate_pct: float = Field(..., ge=0.0, le=10.0)
    is_error_corrected: int = Field(..., ge=0, le=1)

# Health endpoint
@app.get("/heartbeat")
def heartbeat():
    return {"status": "operational", "service": "QuantumBench API"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array([[data.qubit_count,
                              data.gate_depth,
                              data.error_rate_pct,
                              data.is_error_corrected]])

        pred = model.predict(features)[0]

        # ✅ AUTO SAVE JSON
        output = {
            "health_endpoint": "/heartbeat",
            "predict_endpoint": "/predict",
            "port": 8500,
            "health_response": {
                "status": "operational",
                "service": "QuantumBench API"
            },
            "test_input": {
                "qubit_count": data.qubit_count,
                "gate_depth": data.gate_depth,
                "error_rate_pct": data.error_rate_pct,
                "is_error_corrected": data.is_error_corrected
            },
            "prediction": float(pred)
        }

        # Ensure folder exists
        os.makedirs("results", exist_ok=True)

        # Save file
        with open("results/step2_s4.json", "w") as f:
            json.dump(output, f, indent=4)

        return {"prediction": float(pred)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))