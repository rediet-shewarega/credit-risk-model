from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

DATE_COL = "TransactionStartTime"
CUSTOMER_ID = "CustomerId"

NUMERIC_FEATURES = [
    "total_amount",
    "avg_amount",
    "std_amount",
    "transaction_count",
    "recency_days",
    "frequency",
    "monetary",
    "fraud_rate",
]
CATEGORICAL_FEATURES = [
    "top_channel",
    "top_product_category",
    "top_pricing_strategy",
    "top_country",
    "top_provider",
]


def _mode_or_missing(series: pd.Series):
    mode = series.mode()
    if mode.empty:
        return "missing"
    return mode.iloc[0]


class TransactionAggregator(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        datetime_col: str = DATE_COL,
        snapshot_date: Optional[str] = None,
    ):
        self.datetime_col = datetime_col
        self.snapshot_date = snapshot_date
        self.snapshot_date_ = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()
        if self.datetime_col not in df.columns:
            raise ValueError(
                f"Missing required datetime column: {self.datetime_col}"
            )

        df[self.datetime_col] = pd.to_datetime(
    df[self.datetime_col],
    errors="coerce",
    utc=True,
)
        if self.snapshot_date is None:
            self.snapshot_date_ = (
                df[self.datetime_col].max() + pd.Timedelta(days=1)
            )
        else:
            self.snapshot_date_ = pd.to_datetime(
    self.snapshot_date,
    utc=True,
)

        grouped = df.groupby(CUSTOMER_ID).agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            std_amount=("Amount", "std"),
            transaction_count=("Amount", "size"),
            first_txn=(self.datetime_col, "min"),
            last_txn=(self.datetime_col, "max"),
            monetary=("Value", "sum"),
            fraud_rate=("FraudResult", "mean"),
            top_channel=("ChannelId", _mode_or_missing),
            top_product_category=(
                "ProductCategory",
                _mode_or_missing,
            ),
            top_pricing_strategy=(
                "PricingStrategy",
                _mode_or_missing,
            ),
            top_country=("CountryCode", _mode_or_missing),
            top_provider=("ProviderId", _mode_or_missing),
        )
        grouped = grouped.reset_index()
        grouped["std_amount"] = (
            grouped["std_amount"].fillna(0.0)
        )
        grouped["fraud_rate"] = (
            grouped["fraud_rate"].fillna(0.0)
        )
        grouped["recency_days"] = (
            self.snapshot_date_ - grouped["last_txn"]
        ).dt.days.clip(lower=0)
        grouped["tenure_days"] = (
            grouped["last_txn"] - grouped["first_txn"]
        ).dt.days.replace(0, 1)
        grouped["frequency"] = (
            grouped["transaction_count"] / grouped["tenure_days"]
        )
        grouped = grouped.drop(
            columns=[
                "first_txn",
                "last_txn",
                "tenure_days",
            ]
        )
        return grouped


def create_feature_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="constant", fill_value="missing"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("preprocessor", preprocessor)])


def build_feature_engineering_pipeline(
    snapshot_date: Optional[str] = None,
) -> Pipeline:
    return Pipeline(
        [
            (
                "aggregator",
                TransactionAggregator(snapshot_date=snapshot_date),
            ),
            ("feature_pipeline", create_feature_pipeline()),
        ]
    )


def compute_rfm_proxy(
    df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    if not {
        "recency_days",
        "frequency",
        "monetary",
    }.issubset(df.columns):
        raise ValueError(
            "RFM features are required to compute the proxy target."
        )

    scaler = StandardScaler()
    rfm = df[["recency_days", "frequency", "monetary"]].copy()
    rfm_scaled = scaler.fit_transform(rfm)

    clusterer = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
    )
    df["cluster"] = clusterer.fit_predict(rfm_scaled)

    cluster_summary = (
        df.groupby("cluster")[
            ["recency_days", "frequency", "monetary"]
        ]
        .mean()
        .sort_values(
            ["frequency", "monetary"], ascending=[True, True]
        )
    )
    risk_cluster = cluster_summary.index[0]
    df["is_high_risk"] = (
        (df["cluster"] == risk_cluster).astype(int)
    )
    return df


def build_model_ready_data(
    raw_data: pd.DataFrame,
    snapshot_date: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series, Pipeline, pd.DataFrame]:
    aggregator = TransactionAggregator(snapshot_date=snapshot_date)
    aggregated = aggregator.fit_transform(raw_data)
    aggregated = compute_rfm_proxy(aggregated)

    feature_pipeline = create_feature_pipeline()
    X = feature_pipeline.fit_transform(aggregated)
    feature_names = (
        feature_pipeline["preprocessor"].get_feature_names_out()
    )
    X_df = pd.DataFrame(
        X,
        columns=feature_names,
        index=aggregated.index,
    )
    y = aggregated["is_high_risk"].astype(int)
    return X_df, y, feature_pipeline, aggregated


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
