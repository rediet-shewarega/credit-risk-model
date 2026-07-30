from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_processing import CUSTOMER_ID, TransactionAggregator

REQUIRED_TRANSACTION_COLUMNS = frozenset(
    {
        CUSTOMER_ID,
        "Amount",
        "Value",
        "TransactionStartTime",
        "ChannelId",
        "ProductCategory",
        "PricingStrategy",
        "CountryCode",
        "ProviderId",
        "FraudResult",
    }
)


@dataclass
class RawTransactionRiskModel:
    """Serializable risk model that accepts raw customer transactions.

    The model aggregates transaction-level input into customer-level
    features and passes those features to a fitted sklearn pipeline.
    """

    customer_pipeline: Pipeline
    snapshot_date: str
    decision_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate model configuration after initialization."""
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError(
                "decision_threshold must be between 0 and 1."
            )

    def _validate_transactions(
        self,
        transactions: pd.DataFrame,
    ) -> None:
        """Validate raw input before running model inference."""
        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame.")

        if transactions.empty:
            raise ValueError("At least one transaction is required.")

        missing_columns = sorted(
            REQUIRED_TRANSACTION_COLUMNS - set(transactions.columns)
        )
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(
                f"Missing required transaction columns: {missing}"
            )

        customer_ids = transactions[CUSTOMER_ID]
        if customer_ids.isna().any():
            raise ValueError("CustomerId cannot be missing.")

        if customer_ids.nunique() != 1:
            raise ValueError(
                "All transactions must belong to one customer."
            )

    def aggregate(
        self,
        transactions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert raw transactions into one customer feature row."""
        self._validate_transactions(transactions)

        aggregator = TransactionAggregator(
            snapshot_date=self.snapshot_date
        )
        return aggregator.transform(transactions)

    def predict_proba(
        self,
        transactions: pd.DataFrame,
    ) -> np.ndarray:
        """Return low-risk and high-risk probabilities."""
        customer_features = self.aggregate(transactions)
        probabilities = self.customer_pipeline.predict_proba(
            customer_features
        )
        probabilities_array = np.asarray(
            probabilities,
            dtype=float,
        )

        if (
            probabilities_array.ndim != 2
            or probabilities_array.shape[1] != 2
        ):
            raise RuntimeError(
                "Classifier must return two probability columns."
            )

        return probabilities_array

    def predict(
        self,
        transactions: pd.DataFrame,
    ) -> np.ndarray:
        """Return binary risk labels using the configured threshold."""
        probabilities = self.predict_proba(transactions)[:, 1]
        return (
            probabilities >= self.decision_threshold
        ).astype(int)
