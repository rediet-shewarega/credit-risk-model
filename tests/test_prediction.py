from typing import Any

import numpy as np
import pytest

from src.predict import predict_risk


class StubRiskModel:
    """Small predictable model used to test prediction logic."""

    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities

    def predict_proba(self, data: Any) -> np.ndarray:
        assert len(data) == len(self.probabilities)
        return np.array(
            [[1.0 - probability, probability]
             for probability in self.probabilities]
        )


def test_predict_risk_rejects_empty_transactions() -> None:
    model = StubRiskModel([])

    with pytest.raises(
        ValueError,
        match="At least one transaction is required",
    ):
        predict_risk(model, [])


def test_predict_risk_returns_high_risk_label() -> None:
    model = StubRiskModel([0.80])
    transactions = [{"CustomerId": "C1"}]

    result = predict_risk(model, transactions)

    assert result["risk_probability"] == pytest.approx(0.80)
    assert result["risk_label"] == 1


def test_predict_risk_returns_low_risk_label() -> None:
    model = StubRiskModel([0.20])
    transactions = [{"CustomerId": "C1"}]

    result = predict_risk(model, transactions)

    assert result["risk_probability"] == pytest.approx(0.20)
    assert result["risk_label"] == 0


def test_predict_risk_averages_multiple_probabilities() -> None:
    model = StubRiskModel([0.20, 0.80])
    transactions = [
        {"CustomerId": "C1"},
        {"CustomerId": "C1"},
    ]

    result = predict_risk(model, transactions)

    assert result["risk_probability"] == pytest.approx(0.50)
    assert result["risk_label"] == 1
