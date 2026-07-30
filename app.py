"""Streamlit dashboard for credit-risk portfolio scoring."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard import (
    DashboardConfig,
    MODEL_TRAINING_COMMAND,
    build_risk_distribution,
    build_sample_template_csv,
    calculate_portfolio_kpis,
    load_risk_model,
    resolve_model_path,
    score_customers,
    scored_customers_to_csv,
    validate_dashboard_config,
    validate_transaction_data,
)

PAGE_TITLE = "Credit Risk Portfolio Dashboard"
PAGE_ICON = "🏦"


@st.cache_resource(show_spinner=False)
def _load_cached_model(model_path: str):
    """Load and cache the local risk model artifact."""
    return load_risk_model(model_path)


def _render_model_unavailable(model_path: Path) -> None:
    """Display instructions when the model artifact is unavailable."""
    st.error(
        f"No model artifact was found at `{model_path}`."
    )
    st.markdown(
        """
        This dashboard scores uploaded transaction data with a trained
        **RawTransactionRiskModel** artifact. It does not train models or
        accept uploaded serialized model files.
        """
    )
    st.markdown(
        "Create the model artifact locally with:"
    )
    st.code(MODEL_TRAINING_COMMAND, language="bash")
    st.info(
        "Set the `MODEL_PATH` environment variable to use a different "
        f"artifact location. The default path is `{model_path}`."
    )


def main() -> None:
    """Render the credit-risk portfolio dashboard."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout="wide",
    )

    st.title(PAGE_TITLE)
    st.markdown(
        """
        Monitor customer-level credit risk from raw ecommerce transaction
        uploads. Configure decision thresholds and exposure assumptions in
        the sidebar, then review portfolio KPIs, risk distribution, and
        customer-level scoring outputs.
        """
    )
    st.warning(
        "**Disclaimer:** The model target is an RFM-based proxy label, not "
        "a verified loan default outcome. Use this dashboard to support "
        "underwriting and portfolio monitoring—it does not replace human "
        "credit judgment or formal policy approval."
    )

    model_path = resolve_model_path(os.environ.get("MODEL_PATH"))
    st.sidebar.header("Portfolio Assumptions")
    decision_threshold = st.sidebar.slider(
        "Decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help=(
            "Customers with risk probability at or above this threshold "
            "are classified as high risk."
        ),
    )
    average_exposure = st.sidebar.number_input(
        "Average exposure per customer",
        min_value=1.0,
        value=1000.0,
        step=100.0,
        format="%.2f",
    )
    loss_given_default = st.sidebar.slider(
        "Loss given default (LGD)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.01,
        help="Assumed loss severity if a customer defaults.",
    )

    config = DashboardConfig(
        decision_threshold=decision_threshold,
        average_exposure=average_exposure,
        loss_given_default=loss_given_default,
    )
    try:
        validate_dashboard_config(config)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        st.stop()

    st.sidebar.divider()
    st.sidebar.subheader("Model Artifact")
    st.sidebar.code(str(model_path))

    if not model_path.is_file():
        _render_model_unavailable(model_path)
        st.stop()

    try:
        model = _load_cached_model(str(model_path))
    except (FileNotFoundError, TypeError, ValueError) as exc:
        st.error(f"Unable to load model artifact: {exc}")
        _render_model_unavailable(model_path)
        st.stop()

    metadata_cols = st.columns(3)
    metadata_cols[0].metric(
        "Snapshot date",
        model.snapshot_date,
    )
    metadata_cols[1].metric(
        "Stored model threshold",
        f"{model.decision_threshold:.2f}",
    )
    metadata_cols[2].metric(
        "Active dashboard threshold",
        f"{config.decision_threshold:.2f}",
    )

    st.subheader("Upload Transaction Data")
    st.markdown(
        "Upload a CSV containing raw transaction records. Each row must "
        "follow the schema required by `RawTransactionRiskModel`."
    )

    template_csv = build_sample_template_csv()
    st.download_button(
        label="Download sample CSV template",
        data=template_csv,
        file_name="credit_risk_transaction_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Transaction CSV",
        type=["csv"],
        help=(
            "Only raw transaction CSV files are accepted. Model artifacts "
            "cannot be uploaded for security reasons."
        ),
    )

    if uploaded_file is None:
        st.info(
            "Upload a transaction CSV to generate portfolio scores."
        )
        st.stop()

    try:
        transactions = pd.read_csv(uploaded_file)
        validate_transaction_data(transactions)
    except ValueError as exc:
        st.error(f"Data validation failed: {exc}")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("Uploaded CSV is empty.")
        st.stop()
    except Exception as exc:
        st.error(f"Unable to read uploaded CSV: {exc}")
        st.stop()

    try:
        scored_customers = score_customers(
            transactions,
            model,
            config,
        )
        kpis = calculate_portfolio_kpis(scored_customers, config)
    except ValueError as exc:
        st.error(f"Scoring failed: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected scoring error: {exc}")
        st.stop()

    st.subheader("Portfolio KPIs")
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Customers scored", f"{kpis.customer_count:,}")
    kpi_cols[1].metric("High-risk customers", f"{kpis.high_risk_count:,}")
    kpi_cols[2].metric(
        "High-risk rate",
        f"{kpis.high_risk_rate:.1%}",
    )

    kpi_cols = st.columns(3)
    kpi_cols[0].metric(
        "Average risk probability",
        f"{kpis.average_risk_probability:.1%}",
    )
    kpi_cols[1].metric(
        "Total exposure",
        f"${kpis.total_exposure:,.0f}",
    )
    kpi_cols[2].metric(
        "Probability-weighted expected loss",
        f"${kpis.probability_weighted_expected_loss:,.0f}",
    )

    st.subheader("Risk Distribution")
    distribution = build_risk_distribution(scored_customers)
    st.bar_chart(
        distribution.set_index("risk_band")["customer_count"],
    )

    st.subheader("High-Risk Customers")
    high_risk = scored_customers[
        scored_customers["risk_label"] == 1
    ].copy()
    if high_risk.empty:
        st.success(
            "No customers exceed the configured decision threshold."
        )
    else:
        st.dataframe(
            high_risk,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Complete Scored Portfolio")
    st.dataframe(
        scored_customers,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        label="Download scoring results (CSV)",
        data=scored_customers_to_csv(scored_customers),
        file_name="credit_risk_scoring_results.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
