import joblib
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
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
    build_feature_engineering_pipeline,
    build_model_ready_data,
    load_raw_data,
)

MLFLOW_EXPERIMENT_NAME = "credit-risk-scorer"


def train(raw_data_path: str, output_dir: str = "model_artifacts") -> None:
    raw_data = load_raw_data(raw_data_path)
    X, y, _, _ = build_model_ready_data(raw_data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(random_state=42),
    }

    param_grids = {
        "logistic_regression": {"C": [0.1, 1.0, 10.0]},
        "random_forest": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10],
        },
    }

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    best_run = None
    best_score = -1.0

    for name, estimator in models.items():
        grid = GridSearchCV(
            estimator,
            param_grids[name],
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        with mlflow.start_run(run_name=name):
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("train_size", len(X_train))
            mlflow.log_metric("test_size", len(X_test))
            mlflow.sklearn.log_model(
                grid.best_estimator_,
                artifact_path="model",
            )

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_run = {
                "name": name,
                "estimator": grid.best_estimator_,
                "metrics": metrics,
                "params": grid.best_params_,
            }

    if best_run is None:
        raise RuntimeError("No models were successfully trained.")

    os.makedirs(output_dir, exist_ok=True)
    full_pipeline = Pipeline(
        [
            ("transform", build_feature_engineering_pipeline()),
            ("classifier", best_run["estimator"]),
        ]
    )
    model_path = Path(output_dir) / "best_model.pkl"
    joblib.dump(full_pipeline, model_path)
    print(f"Best model saved to {model_path}")
