from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.data_processing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TransactionAggregator,
    compute_rfm_proxy,
    create_feature_pipeline,
    load_raw_data,
)
from src.model import RawTransactionRiskModel

MLFLOW_EXPERIMENT_NAME = "credit-risk-scorer"
MODEL_FILENAME = "best_model.pkl"

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable configuration for the credit-risk training workflow."""

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 3
    scoring: str = "roc_auc"
    decision_threshold: float = 0.5
    snapshot_date: str | None = None


def _resolve_snapshot_date(aggregator: TransactionAggregator) -> str:
    """Return the snapshot date resolved during transaction aggregation."""
    if aggregator.snapshot_date_ is None:
        raise RuntimeError(
            "Snapshot date was not resolved during aggregation."
        )
    return aggregator.snapshot_date_.strftime("%Y-%m-%d")


def _build_candidate_pipelines(
    config: TrainingConfig,
) -> dict[str, Pipeline]:
    """Build sklearn pipelines with feature engineering and classifiers."""
    return {
        "logistic_regression": Pipeline(
            [
                ("features", create_feature_pipeline()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        random_state=config.random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("features", create_feature_pipeline()),
                (
                    "classifier",
                    RandomForestClassifier(
                        random_state=config.random_state,
                    ),
                ),
            ]
        ),
    }


def _param_grids() -> dict[str, dict[str, list[float | int]]]:
    """Return GridSearchCV parameter grids keyed by model name."""
    return {
        "logistic_regression": {
            "classifier__C": [0.1, 1.0, 10.0],
        },
        "random_forest": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [5, 10],
        },
    }


def _extract_features_and_target(
    aggregated: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split aggregated customer data into model features and target."""
    features = aggregated[FEATURE_COLUMNS]
    target = aggregated["is_high_risk"].astype(int)
    return features, target


def _evaluate_classifier(
    y_test: pd.Series,
    y_pred: pd.Series,
    y_proba: pd.Series,
) -> dict[str, float]:
    """Compute classification metrics on the held-out test set."""
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def _prepare_training_data(
    raw_data_path: str | Path,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, str]:
    """Load, aggregate, label, and split customer-level training data."""
    raw_data = load_raw_data(str(raw_data_path))
    aggregator = TransactionAggregator(
        snapshot_date=config.snapshot_date,
    )
    aggregated = aggregator.fit_transform(raw_data)
    aggregated = compute_rfm_proxy(
        aggregated,
        random_state=config.random_state,
    )
    snapshot_date = _resolve_snapshot_date(aggregator)

    train_df, test_df = train_test_split(
        aggregated,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=aggregated["is_high_risk"],
    )
    x_train, y_train = _extract_features_and_target(train_df)
    x_test, y_test = _extract_features_and_target(test_df)
    return x_train, y_train, x_test, y_test, snapshot_date


def train(
    raw_data_path: str | Path,
    output_dir: str | Path = "model_artifacts",
    config: TrainingConfig | None = None,
) -> Path:
    """Train candidate models, log runs to MLflow, and save the best scorer."""
    training_config = config or TrainingConfig()
    x_train, y_train, x_test, y_test, snapshot_date = _prepare_training_data(
        raw_data_path,
        training_config,
    )

    candidate_pipelines = _build_candidate_pipelines(training_config)
    param_grids = _param_grids()

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    best_run: dict[str, object] | None = None
    best_score = -1.0

    for name, pipeline in candidate_pipelines.items():
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=training_config.cv_folds,
            scoring=training_config.scoring,
            n_jobs=-1,
        )
        grid_search.fit(x_train, y_train)

        fitted_pipeline = grid_search.best_estimator_
        y_pred = fitted_pipeline.predict(x_test)
        y_proba = fitted_pipeline.predict_proba(x_test)[:, 1]
        metrics = _evaluate_classifier(y_test, y_pred, y_proba)

        with mlflow.start_run(run_name=name):
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("train_size", len(x_train))
            mlflow.log_metric("test_size", len(x_test))
            mlflow.sklearn.log_model(
                fitted_pipeline,
                name="model",
                serialization_format=(
                    mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
                ),
            )

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_run = {
                "name": name,
                "pipeline": fitted_pipeline,
                "metrics": metrics,
                "params": grid_search.best_params_,
            }

    if best_run is None:
        raise RuntimeError("No models were successfully trained.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fitted_pipeline = best_run["pipeline"]
    assert isinstance(fitted_pipeline, Pipeline)

    risk_model = RawTransactionRiskModel(
        customer_pipeline=fitted_pipeline,
        snapshot_date=snapshot_date,
        decision_threshold=training_config.decision_threshold,
    )

    model_path = output_path / MODEL_FILENAME
    joblib.dump(risk_model, model_path)
    print(f"Best model saved to {model_path}")
    return model_path


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training CLI."""
    parser = argparse.ArgumentParser(
        description="Train credit-risk models from raw transaction data.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the raw transaction CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_artifacts",
        help="Directory where the best model artifact will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run model training from the command line."""
    args = _parse_args()
    train(args.data, args.output_dir)


if __name__ == "__main__":
    main()
