# Credit Risk Probability Model for Alternative Data

![CI](https://github.com/rediet-shewarega/credit-risk-model/actions/workflows/ci.yml/badge.svg?branch=week12-production-upgrade)

This project implements an end-to-end credit risk scoring solution for Bati Bank using alternative ecommerce transaction data.

## Business Understanding

### Basel II and Interpretable Modeling
Basel II emphasizes that credit risk models must be documented, auditable, and stable over time. For Bati Bank, this means the model should prioritize explainability and reproducibility. Interpretable models and feature transformations make it easier to justify risk decisions to regulators and to embed model governance controls.

### Why a Proxy Target Variable Is Necessary
The raw dataset has transaction-level behavior but no direct default outcome. A proxy target is therefore necessary to approximate credit stress using behavioral signals such as Recency, Frequency, and Monetary patterns. This proxy-based approach introduces business risk because it is an indirect label: it can capture engagement and payment propensity but may not perfectly represent actual loan defaults.

### Trade-offs Between Interpretability and Performance
A simple model like Logistic Regression with Weight of Evidence (WoE) is easier to explain, validate, and maintain under Basel II. It supports stable decision rules and consistent documentation. A higher-capacity model like gradient boosting can yield stronger predictive performance, but it increases the burden of transparency, monitoring, and regulatory explanation.

## Project Structure

- `data/` — raw and processed data storage (ignored by git)
- `notebooks/` — exploratory analysis and EDA
- `src/` — production code for data processing, training, prediction, and API
- `tests/` — unit tests
- `Dockerfile` — container image for the scoring service
- `docker-compose.yml` — local service orchestration
- `.github/workflows/ci.yml` — CI pipeline for linting and tests

## Credit Scoring Workflow

1. Load ecommerce transaction records.
2. Create customer-level RFM and aggregated features.
3. Define a proxy default target using RFM-based clustering.
4. Train candidate classifiers while tracking experiments.
5. Serve the best model through a FastAPI endpoint.
