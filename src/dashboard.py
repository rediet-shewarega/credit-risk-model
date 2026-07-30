"""Framework-independent logic for the credit-risk Streamlit dashboard."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd

from src.data_processing import CUSTOMER_ID
from src.model import REQUIRED_TRANSACTION_COLUMNS, RawTransactionRiskModel

NUMERIC_TRANSACTION_COLUMNS = frozenset({"Amount", "Value", "FraudResult"})

SCORED_CUSTOMER_COLUMNS = [
    "CustomerId",
    "risk_probability",
    "risk_label",
    "transaction_count",
    "total_transaction_amount",
]

MODEL_TRAINING_COMMAND = (
    "python -m src.train --data <path-to-csv> --output-dir model_artifacts"
)

DEFAULT_MODEL_PATH = Path("model_artifacts/best_model.pkl")

SAMPLE_TEMPLATE_ROWS = [
    {
        "CustomerId": "CUST-001",
        "Amount": 250.0,
        "Value": 250.0,
        "TransactionStartTime": "2025-12-01T10:00:00",
        "ChannelId": "web",
        "ProductCategory": "electronics",
        "PricingStrategy": "standard",
        "CountryCode": 1,
        "ProviderId": 10,
        "FraudResult": 0,
    },
    {
        "CustomerId": "CUST-001",
        "Amount": 150.0,
        "Value": 150.0,
        "TransactionStartTime": "2025-12-02T14:30:00",
        "ChannelId": "mobile",
        "ProductCategory": "electronics",
        "PricingStrategy": "standard",
        "CountryCode": 1,
        "ProviderId": 10,
        "FraudResult": 0,
    },
]


class RiskModelProtocol(Protocol):
    """Minimal interface required to score customer transactions."""

    def predict_proba(self, transactions: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for one customer's transactions."""


@dataclass(frozen=True)
class DashboardConfig:
    """Immutable dashboard parameters for scoring and portfolio analytics."""

    decision_threshold: float = 0.5
    average_exposure: float = 1000.0
    loss_given_default: float = 0.45


@dataclass(frozen=True)
class PortfolioKPIs:
    """Summary metrics for a scored customer portfolio."""

    customer_count: int
    high_risk_count: int
    high_risk_rate: float
    average_risk_probability: float
    total_exposure: float
    probability_weighted_expected_loss: float


def validate_dashboard_config(config: DashboardConfig) -> None:
    """Validate dashboard configuration values.

    Args:
        config: Dashboard parameters to validate.

    Raises:
        ValueError: If any configuration value is outside allowed bounds.
    """
    if not 0.0 < config.decision_threshold < 1.0:
        raise ValueError(
            "Decision threshold must be strictly between 0 and 1."
        )

    if config.average_exposure <= 0.0:
        raise ValueError(
            "Average exposure must be greater than zero."
        )

    if not 0.0 <= config.loss_given_default <= 1.0:
        raise ValueError(
            "Loss given default must be between 0 and 1 inclusive."
        )


def validate_transaction_data(transactions: pd.DataFrame) -> None:
    """Validate uploaded raw transaction data for portfolio scoring.

    Args:
        transactions: Raw transaction records uploaded by the user.

    Raises:
        TypeError: If the input is not a pandas DataFrame.
        ValueError: If required columns, values, or types are invalid.
    """
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("Transaction data must be a pandas DataFrame.")

    if transactions.empty:
        raise ValueError("Transaction data cannot be empty.")

    missing_columns = sorted(
        REQUIRED_TRANSACTION_COLUMNS - set(transactions.columns)
    )
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"Missing required transaction columns: {missing}"
        )

    for column in sorted(REQUIRED_TRANSACTION_COLUMNS):
        if transactions[column].isna().any():
            raise ValueError(
                f"Column '{column}' contains missing values."
            )

    for column in sorted(NUMERIC_TRANSACTION_COLUMNS):
        numeric_values = pd.to_numeric(
            transactions[column],
            errors="coerce",
        )
        if numeric_values.isna().any():
            raise ValueError(
                f"Column '{column}' contains invalid numeric values."
            )

    parsed_dates = pd.to_datetime(
        transactions["TransactionStartTime"],
        errors="coerce",
    )
    if parsed_dates.isna().any():
        raise ValueError(
            "Column 'TransactionStartTime' contains invalid datetime "
            "values."
        )


def load_risk_model(model_path: str | Path) -> RawTransactionRiskModel:
    """Load a serialized risk model from a local artifact path.

    Args:
        model_path: Filesystem path to a joblib-serialized model artifact.

    Returns:
        The loaded ``RawTransactionRiskModel`` instance.

    Raises:
        FileNotFoundError: If the model artifact does not exist.
        TypeError: If the artifact does not deserialize to the expected type.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'."
        )

    loaded_model = joblib.load(path)
    if not isinstance(loaded_model, RawTransactionRiskModel):
        raise TypeError(
            "Model artifact must deserialize to RawTransactionRiskModel."
        )

    return loaded_model


def resolve_model_path(
    model_path: str | Path | None = None,
) -> Path:
    """Resolve the configured model artifact path.

    Args:
        model_path: Optional explicit model path override.

    Returns:
        The resolved filesystem path to the model artifact.
    """
    if model_path is not None:
        return Path(model_path)
    return DEFAULT_MODEL_PATH


def build_sample_template_csv() -> str:
    """Return a CSV string containing the required upload schema.

    Returns:
        CSV content for a downloadable sample transaction template.
    """
    sample = pd.DataFrame(SAMPLE_TEMPLATE_ROWS)
    return sample.to_csv(index=False)


def score_customers(
    transactions: pd.DataFrame,
    model: RiskModelProtocol,
    config: DashboardConfig,
) -> pd.DataFrame:
    """Score each unique customer from raw transaction records.

    Args:
        transactions: Validated raw transaction data for many customers.
        model: Fitted risk model that accepts one customer's transactions.
        config: Dashboard parameters including the decision threshold.

    Returns:
        Customer-level scoring results sorted by descending risk probability.

    Raises:
        ValueError: If validation fails or scoring cannot be completed.
    """
    validate_dashboard_config(config)
    validate_transaction_data(transactions)

    scored_rows: list[dict[str, object]] = []
    for customer_id, customer_transactions in transactions.groupby(
        CUSTOMER_ID,
        sort=True,
    ):
        try:
            probabilities = model.predict_proba(customer_transactions)
        except Exception as exc:
            raise ValueError(
                f"Failed to score customer '{customer_id}': {exc}"
            ) from exc

        probability_array = np.asarray(probabilities, dtype=float)
        if (
            probability_array.ndim != 2
            or probability_array.shape[0] != 1
            or probability_array.shape[1] != 2
        ):
            raise ValueError(
                f"Unexpected probability output for customer '{customer_id}'."
            )

        risk_probability = float(probability_array[0, 1])
        risk_label = int(
            risk_probability >= config.decision_threshold
        )
        scored_rows.append(
            {
                "CustomerId": customer_id,
                "risk_probability": risk_probability,
                "risk_label": risk_label,
                "transaction_count": int(len(customer_transactions)),
                "total_transaction_amount": float(
                    pd.to_numeric(
                        customer_transactions["Amount"],
                        errors="coerce",
                    ).sum()
                ),
            }
        )

    scored = pd.DataFrame(scored_rows, columns=SCORED_CUSTOMER_COLUMNS)
    return scored.sort_values(
        by=["risk_probability", "CustomerId"],
        ascending=[False, True],
    ).reset_index(drop=True)


def calculate_portfolio_kpis(
    scored_customers: pd.DataFrame,
    config: DashboardConfig,
) -> PortfolioKPIs:
    """Calculate portfolio-level KPIs from customer scoring results.

    Args:
        scored_customers: Output from :func:`score_customers`.
        config: Dashboard parameters used for exposure and loss assumptions.

    Returns:
        Portfolio KPI summary metrics.

    Raises:
        ValueError: If the scored portfolio is empty or config is invalid.
    """
    validate_dashboard_config(config)

    if scored_customers.empty:
        raise ValueError(
            "Scored customer portfolio cannot be empty."
        )

    required_columns = set(SCORED_CUSTOMER_COLUMNS)
    missing_columns = sorted(
        required_columns - set(scored_customers.columns)
    )
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"Scored customer data is missing columns: {missing}"
        )

    customer_count = int(len(scored_customers))
    high_risk_count = int(scored_customers["risk_label"].sum())
    high_risk_rate = (
        high_risk_count / customer_count if customer_count else 0.0
    )
    average_risk_probability = float(
        scored_customers["risk_probability"].mean()
    )
    total_exposure = customer_count * config.average_exposure
    probability_weighted_expected_loss = float(
        (
            scored_customers["risk_probability"]
            * config.average_exposure
            * config.loss_given_default
        ).sum()
    )

    return PortfolioKPIs(
        customer_count=customer_count,
        high_risk_count=high_risk_count,
        high_risk_rate=high_risk_rate,
        average_risk_probability=average_risk_probability,
        total_exposure=total_exposure,
        probability_weighted_expected_loss=(
            probability_weighted_expected_loss
        ),
    )


def build_risk_distribution(scored_customers: pd.DataFrame) -> pd.DataFrame:
    """Build a binned risk distribution suitable for charting.

    Args:
        scored_customers: Customer-level scoring results.

    Returns:
        DataFrame with ``risk_band`` and ``customer_count`` columns.
    """
    if scored_customers.empty:
        return pd.DataFrame(
            columns=["risk_band", "customer_count"],
        )

    bands = pd.cut(
        scored_customers["risk_probability"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[
            "0-20%",
            "20-40%",
            "40-60%",
            "60-80%",
            "80-100%",
        ],
        include_lowest=True,
        right=True,
    )
    distribution = (
        bands.value_counts(sort=False)
        .rename_axis("risk_band")
        .reset_index(name="customer_count")
    )
    return distribution


def scored_customers_to_csv(scored_customers: pd.DataFrame) -> str:
    """Serialize scored customer results to CSV text.

    Args:
        scored_customers: Customer-level scoring results.

    Returns:
        CSV string suitable for download.
    """
    buffer = io.StringIO()
    scored_customers.to_csv(buffer, index=False)
    return buffer.getvalue()
