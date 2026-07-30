"""Unit tests for dashboard scoring and portfolio analytics."""

from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import pytest

from src.dashboard import (
    DashboardConfig,
    build_sample_template_csv,
    calculate_portfolio_kpis,
    score_customers,
    validate_dashboard_config,
    validate_transaction_data,
)


class StubRiskModel:
    """Deterministic model stub keyed by customer identifier."""

    def __init__(self, customer_probs: dict[str, float]) -> None:
        self.customer_probs = customer_probs

    def predict_proba(self, transactions: pd.DataFrame) -> np.ndarray:
        customer_id = str(transactions["CustomerId"].iloc[0])
        probability = self.customer_probs[customer_id]
        return np.array([[1.0 - probability, probability]])


def _transaction_row(
    customer_id: str,
    amount: float,
    timestamp: str,
) -> dict[str, object]:
    return {
        "CustomerId": customer_id,
        "Amount": amount,
        "Value": amount,
        "TransactionStartTime": timestamp,
        "ChannelId": "web",
        "ProductCategory": "electronics",
        "PricingStrategy": "standard",
        "CountryCode": 1,
        "ProviderId": 10,
        "FraudResult": 0,
    }


def _sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _transaction_row("C1", 100.0, "2025-12-01T10:00:00"),
            _transaction_row("C1", 50.0, "2025-12-02T10:00:00"),
            _transaction_row("C2", 200.0, "2025-12-01T11:00:00"),
            _transaction_row("C3", 75.0, "2025-12-03T09:00:00"),
            _transaction_row("C3", 25.0, "2025-12-04T09:00:00"),
        ]
    )


def test_score_customers_across_multiple_customers() -> None:
    transactions = _sample_transactions()
    model = StubRiskModel({"C1": 0.80, "C2": 0.30, "C3": 0.55})
    config = DashboardConfig(decision_threshold=0.50)

    scored = score_customers(transactions, model, config)

    assert list(scored["CustomerId"]) == ["C1", "C3", "C2"]
    assert scored.loc[scored["CustomerId"] == "C1", "risk_probability"].iloc[
        0
    ] == pytest.approx(0.80)
    assert scored.loc[scored["CustomerId"] == "C1", "risk_label"].iloc[
        0
    ] == 1
    assert scored.loc[scored["CustomerId"] == "C2", "risk_label"].iloc[
        0
    ] == 0
    assert scored.loc[
        scored["CustomerId"] == "C1", "transaction_count"
    ].iloc[0] == 2
    assert scored.loc[
        scored["CustomerId"] == "C1", "total_transaction_amount"
    ].iloc[0] == pytest.approx(150.0)


def test_validate_transaction_data_requires_columns() -> None:
    transactions = _sample_transactions().drop(columns=["Amount"])

    with pytest.raises(ValueError, match="Missing required transaction"):
        validate_transaction_data(transactions)


def test_validate_transaction_data_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_transaction_data(pd.DataFrame())


def test_threshold_behavior_changes_risk_labels() -> None:
    transactions = _sample_transactions().query("CustomerId == 'C3'")
    model = StubRiskModel({"C3": 0.55})

    low_threshold = score_customers(
        transactions,
        model,
        DashboardConfig(decision_threshold=0.60),
    )
    high_threshold = score_customers(
        transactions,
        model,
        DashboardConfig(decision_threshold=0.50),
    )

    assert low_threshold.loc[0, "risk_label"] == 0
    assert high_threshold.loc[0, "risk_label"] == 1


def test_calculate_portfolio_kpis() -> None:
    scored = pd.DataFrame(
        [
            {
                "CustomerId": "C1",
                "risk_probability": 0.80,
                "risk_label": 1,
                "transaction_count": 2,
                "total_transaction_amount": 150.0,
            },
            {
                "CustomerId": "C2",
                "risk_probability": 0.20,
                "risk_label": 0,
                "transaction_count": 1,
                "total_transaction_amount": 200.0,
            },
        ]
    )
    config = DashboardConfig(
        average_exposure=1000.0,
        loss_given_default=0.50,
    )

    kpis = calculate_portfolio_kpis(scored, config)

    assert kpis.customer_count == 2
    assert kpis.high_risk_count == 1
    assert kpis.high_risk_rate == pytest.approx(0.5)
    assert kpis.average_risk_probability == pytest.approx(0.5)
    assert kpis.total_exposure == pytest.approx(2000.0)
    assert kpis.probability_weighted_expected_loss == pytest.approx(
        (0.80 + 0.20) * 1000.0 * 0.50
    )


def test_validate_dashboard_config_rejects_invalid_exposure_and_lgd() -> None:
    with pytest.raises(ValueError, match="Average exposure"):
        validate_dashboard_config(
            DashboardConfig(average_exposure=0.0),
        )

    with pytest.raises(ValueError, match="Loss given default"):
        validate_dashboard_config(
            DashboardConfig(loss_given_default=1.5),
        )


def test_build_sample_template_csv_contains_required_columns() -> None:
    template = pd.read_csv(StringIO(build_sample_template_csv()))

    assert not template.empty
    validate_transaction_data(template)
