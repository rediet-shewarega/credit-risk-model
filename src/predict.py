import joblib
from typing import Any

import pandas as pd


def load_model(model_path: str):
    return joblib.load(model_path)


def predict_risk(
    model: Any,
    transactions: list[dict],
) -> dict[str, Any]:
    if not transactions:
        raise ValueError(
            "At least one transaction is required for prediction."
        )

    df = pd.DataFrame(transactions)
    probabilities = model.predict_proba(df)
    if probabilities.ndim == 2:
        probability = float(probabilities[:, 1].mean())
    else:
        probability = float(probabilities[0])
    is_high_risk = int(probability >= 0.5)
    return {
        "risk_probability": probability,
        "risk_label": is_high_risk,
    }
