import os
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictionRequest, PredictionResponse
from src.predict import load_model, predict_risk

app = FastAPI(title="Credit Risk Scoring API")

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/app/model_artifacts/best_model.pkl"
)
model = None


@app.on_event("startup")
def load_scoring_model():
    global model
    try:
        model = load_model(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {exc}")


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    try:
        records = [transaction.dict() for transaction in request.transactions]
        result = predict_risk(model, records)
        return PredictionResponse(
            customer_id=request.transactions[0].CustomerId,
            risk_probability=result["risk_probability"],
            risk_label=result["risk_label"],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
