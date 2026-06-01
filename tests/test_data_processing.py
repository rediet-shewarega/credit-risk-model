import pandas as pd

from src.data_processing import TransactionAggregator, compute_rfm_proxy


def test_transaction_aggregator_produces_expected_columns():
    df = pd.DataFrame(
        [
            {
                "CustomerId": "C1",
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
                "CustomerId": "C1",
                "Amount": 50.0,
                "Value": 50.0,
                "TransactionStartTime": "2025-12-02T12:00:00",
                "ChannelId": "web",
                "ProductCategory": "electronics",
                "PricingStrategy": "standard",
                "CountryCode": 1,
                "ProviderId": 10,
                "FraudResult": 0,
            },
        ]
    )

    aggregator = TransactionAggregator(snapshot_date="2025-12-03")
    result = aggregator.transform(df)

    expected_columns = {
        "CustomerId",
        "total_amount",
        "avg_amount",
        "std_amount",
        "transaction_count",
        "recency_days",
        "frequency",
        "monetary",
        "fraud_rate",
        "top_channel",
        "top_product_category",
        "top_pricing_strategy",
        "top_country",
        "top_provider",
    }
    assert expected_columns.issubset(set(result.columns))
    assert result.loc[0, "transaction_count"] == 2
    assert result.loc[0, "total_amount"] == 150.0


def test_compute_rfm_proxy_assigns_high_risk_label():
    df = pd.DataFrame(
        [
            {
                "CustomerId": "C1",
                "total_amount": 500.0,
                "avg_amount": 100.0,
                "std_amount": 10.0,
                "transaction_count": 5,
                "recency_days": 5,
                "frequency": 0.5,
                "monetary": 500.0,
                "fraud_rate": 0.0,
                "top_channel": "web",
                "top_product_category": "electronics",
                "top_pricing_strategy": "standard",
                "top_country": 1,
                "top_provider": 10,
            },
            {
                "CustomerId": "C2",
                "total_amount": 50.0,
                "avg_amount": 50.0,
                "std_amount": 0.0,
                "transaction_count": 1,
                "recency_days": 60,
                "frequency": 0.02,
                "monetary": 50.0,
                "fraud_rate": 0.0,
                "top_channel": "mobile",
                "top_product_category": "books",
                "top_pricing_strategy": "discount",
                "top_country": 2,
                "top_provider": 20,
            },
            {
                "CustomerId": "C3",
                "total_amount": 450.0,
                "avg_amount": 150.0,
                "std_amount": 20.0,
                "transaction_count": 3,
                "recency_days": 8,
                "frequency": 0.4,
                "monetary": 450.0,
                "fraud_rate": 0.0,
                "top_channel": "web",
                "top_product_category": "fashion",
                "top_pricing_strategy": "standard",
                "top_country": 1,
                "top_provider": 11,
            },
        ]
    )

    result = compute_rfm_proxy(df, n_clusters=3, random_state=42)
    assert "is_high_risk" in result.columns
    assert set(result["is_high_risk"].unique()).issubset({0, 1})
    high_risk_value = result.loc[
        result["CustomerId"] == "C2", "is_high_risk"
    ].iloc[0]
    assert high_risk_value == 1
