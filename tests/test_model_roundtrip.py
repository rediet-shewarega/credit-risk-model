from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_processing import create_feature_pipeline
from src.model import RawTransactionRiskModel


def _feature_row(
    total: float,
    count: int,
    recency: int,
    channel: str,
) -> dict[str, object]:
    return {
        "total_amount": total,
        "avg_amount": total / count,
        "std_amount": 10.0,
        "transaction_count": count,
        "recency_days": recency,
        "frequency": count / max(recency, 1),
        "monetary": abs(total),
        "fraud_rate": 0.0,
        "top_channel": channel,
        "top_product_category": "electronics",
        "top_pricing_strategy": "standard",
        "top_country": 1,
        "top_provider": 10,
    }


def _training_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _feature_row(500.0, 5, 5, "web"),
            _feature_row(450.0, 4, 8, "web"),
            _feature_row(600.0, 6, 3, "web"),
            _feature_row(50.0, 1, 60, "mobile"),
            _feature_row(70.0, 1, 45, "mobile"),
            _feature_row(100.0, 2, 30, "mobile"),
        ]
    )


def _raw_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "CustomerId": "C100",
                "Amount": 100.0,
                "Value": 100.0,
                "TransactionStartTime": "2025-12-01T10:00:00",
                "ChannelId": "web",
                "ProductCategory": "electronics",
                "PricingStrategy": "standard",
                "CountryCode": 1,
                "ProviderId": 10,
                "FraudResult": 0,
            },
            {
                "CustomerId": "C100",
                "Amount": 50.0,
                "Value": 50.0,
                "TransactionStartTime": "2025-12-02T10:00:00",
                "ChannelId": "web",
                "ProductCategory": "electronics",
                "PricingStrategy": "standard",
                "CountryCode": 1,
                "ProviderId": 10,
                "FraudResult": 0,
            },
        ]
    )


def test_serialized_model_accepts_raw_transactions(
    tmp_path: Path,
) -> None:
    features = _training_features()
    labels = pd.Series([0, 0, 0, 1, 1, 1])

    customer_pipeline = Pipeline(
        [
            ("features", create_feature_pipeline()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    customer_pipeline.fit(features, labels)

    scorer = RawTransactionRiskModel(
        customer_pipeline=customer_pipeline,
        snapshot_date="2025-12-31",
    )
    raw_transactions = _raw_transactions()
    expected = scorer.predict_proba(raw_transactions)

    model_path = tmp_path / "credit_risk_model.joblib"
    joblib.dump(scorer, model_path)
    loaded_scorer = joblib.load(model_path)
    actual = loaded_scorer.predict_proba(raw_transactions)

    assert expected.shape == (1, 2)
    np.testing.assert_allclose(actual, expected)
