from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.model import RawTransactionRiskModel
from src.train import (
    MODEL_FILENAME,
    TrainingConfig,
    _build_candidate_pipelines,
    _param_grids,
    train,
)


def test_training_config_defaults() -> None:
    config = TrainingConfig()

    assert config.test_size == 0.2
    assert config.random_state == 42
    assert config.cv_folds == 3
    assert config.scoring == "roc_auc"
    assert config.decision_threshold == 0.5
    assert config.snapshot_date is None


def test_candidate_pipelines_contain_features_and_classifier() -> None:
    config = TrainingConfig()
    pipelines = _build_candidate_pipelines(config)

    assert set(pipelines) == {"logistic_regression", "random_forest"}
    for pipeline in pipelines.values():
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["features", "classifier"]


def test_param_grids_use_classifier_prefixes() -> None:
    param_grids = _param_grids()

    logistic_params = param_grids["logistic_regression"]
    forest_params = param_grids["random_forest"]

    assert set(logistic_params) == {"classifier__C"}
    assert set(forest_params) == {
        "classifier__n_estimators",
        "classifier__max_depth",
    }
    for params in param_grids.values():
        for key in params:
            assert key.startswith("classifier__")


def _synthetic_transaction(
    customer_id: str,
    amount: float,
    value: float,
    timestamp: str,
    channel: str,
) -> dict[str, object]:
    return {
        "CustomerId": customer_id,
        "Amount": amount,
        "Value": value,
        "TransactionStartTime": timestamp,
        "ChannelId": channel,
        "ProductCategory": "electronics",
        "PricingStrategy": "standard",
        "CountryCode": 1,
        "ProviderId": 10,
        "FraudResult": 0,
    }


def _write_synthetic_training_csv(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for index in range(30):
        customer_id = f"C{index:03d}"
        if index >= 20:
            amounts = [20.0, 25.0]
            timestamps = [
                "2025-11-01T10:00:00",
                "2025-11-02T10:00:00",
            ]
            channel = "mobile"
        elif index >= 10:
            amounts = [60.0, 70.0]
            timestamps = [
                "2025-11-20T10:00:00",
                "2025-11-25T10:00:00",
            ]
            channel = "branch"
        else:
            amounts = [120.0, 150.0, 130.0]
            timestamps = [
                "2025-12-01T10:00:00",
                "2025-12-02T10:00:00",
                "2025-12-03T10:00:00",
            ]
            channel = "web"

        for amount, timestamp in zip(amounts, timestamps):
            rows.append(
                _synthetic_transaction(
                    customer_id,
                    amount,
                    amount,
                    timestamp,
                    channel,
                )
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def _prediction_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _synthetic_transaction(
                "C000",
                100.0,
                100.0,
                "2025-12-04T10:00:00",
                "web",
            ),
            _synthetic_transaction(
                "C000",
                50.0,
                50.0,
                "2025-12-05T10:00:00",
                "web",
            ),
        ]
    )


@pytest.fixture
def offline_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tracking_uri = tmp_path / "mlruns"
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tracking_uri}")
    mlflow.set_tracking_uri(f"file:{tracking_uri}")
    return tracking_uri


def test_train_saves_loads_and_predicts_from_raw_transactions(
    tmp_path: Path,
    offline_mlflow: Path,
) -> None:
    data_path = tmp_path / "transactions.csv"
    output_dir = tmp_path / "artifacts"
    _write_synthetic_training_csv(data_path)

    config = TrainingConfig(
        snapshot_date="2025-12-31",
        decision_threshold=0.4,
    )
    model_path = train(
        data_path,
        output_dir=output_dir,
        config=config,
    )

    assert model_path == output_dir / MODEL_FILENAME
    assert model_path.name == "best_model.pkl"

    loaded_model = joblib.load(model_path)
    assert isinstance(loaded_model, RawTransactionRiskModel)
    assert loaded_model.snapshot_date == "2025-12-31"
    assert loaded_model.decision_threshold == pytest.approx(0.4)

    raw_transactions = _prediction_transactions()
    probabilities = loaded_model.predict_proba(raw_transactions)
    predictions = loaded_model.predict(raw_transactions)

    assert probabilities.shape == (1, 2)
    assert np.isclose(probabilities.sum(axis=1), 1.0).all()
    assert set(predictions).issubset({0, 1})
