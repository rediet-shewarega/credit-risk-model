# Week 12 Project Selection and Gap Analysis

## Project Selected

**Credit Risk Probability Model for Alternative Transaction Data**

Repository:  
https://github.com/rediet-shewarega/credit-risk-model

## Project Selection Justification

I selected the Credit Risk Probability Model because it is the project most directly aligned with the needs of banks, fintech companies, and other financial institutions. It addresses a practical lending problem: using customer transaction behaviour to estimate credit risk when a verified default label is unavailable.

The original project already contains a strong technical foundation, including RFM-based customer segmentation, feature engineering, multiple machine-learning models, MLflow integration, a FastAPI service, Docker configuration, automated tests, and a GitHub Actions workflow.

However, several components are currently implemented as a scaffold rather than a fully verified production system. Improving this project provides an opportunity to demonstrate the qualities most valued by finance-sector employers: reliability, reproducibility, transparency, risk awareness, model governance, and clear business communication.

## Business Problem

Bati Bank is partnering with an e-commerce company to provide a buy-now-pay-later service. The bank needs a reliable method for assessing customers who may not have conventional credit histories.

The project uses alternative transaction data to:

1. Create a behavioural proxy for high and low credit risk.
2. Estimate a customer's risk probability.
3. Convert model output into an understandable risk category or credit score.
4. Support transparent lending decisions.
5. Demonstrate how risk-based loan policies could be implemented.

Because the dataset does not contain an observed loan-default outcome, the model's target is a behavioural proxy rather than ground truth. Therefore, all results must be presented as decision-support evidence and not as a validated production probability of default.

## Original Project Accomplishments

The original project includes:

- A standardized and modular repository structure.
- Business understanding based on Basel II considerations.
- An exploratory-analysis notebook.
- Customer-level transaction aggregation.
- Recency, Frequency, and Monetary feature engineering.
- K-Means clustering for proxy-target creation.
- Numerical imputation and standardization.
- Categorical imputation and one-hot encoding.
- Logistic Regression and Random Forest candidate models.
- GridSearchCV hyperparameter tuning.
- Accuracy, precision, recall, F1, and ROC-AUC evaluation.
- MLflow experiment-logging code.
- A FastAPI prediction endpoint.
- Pydantic request and response schemas.
- Docker and Docker Compose configuration.
- A GitHub Actions workflow using flake8 and pytest.
- Six locally passing unit tests after the Week 12 testing improvement.

## Week 12 Gap Analysis

| Category | Question | Status | Evidence and Gap |
|---|---|---|---|
| Code Quality | Is the code modular and well organized? | Partial | Production logic is separated into data processing, training, prediction, and API modules. Some modules still contain multiple responsibilities. |
| Code Quality | Are type hints included on all functions? | Partial | Several functions have type hints, but return types and framework functions are not consistently annotated. |
| Code Quality | Is there a clear project structure? | Yes | The repository contains `src`, `tests`, `notebooks`, API, Docker, CI, and documentation files. |
| Testing | Are there unit tests for core functions? | Yes | Six unit tests currently pass locally. |
| Testing | Do tests run automatically on push? | Partial | GitHub Actions is configured for pushes to `main`, but the Week 12 branch has not yet been pushed and verified remotely. |
| Documentation | Is the README comprehensive? | Partial | The README explains the business context and workflow but lacks verified results, complete setup instructions, screenshots, demo, limitations, and author information. |
| Documentation | Are there docstrings on functions? | No | Most production functions and classes do not have complete professional docstrings. |
| Reproducibility | Can another person run the project? | Partial | Dependencies install and tests pass, but model training, artifact generation, API startup, and Docker execution have not been demonstrated end to end. |
| Reproducibility | Are dependencies documented? | Yes | Dependencies are included in `requirements.txt`, although version reproducibility can be strengthened. |
| Visualization | Is there an interactive way to explore results? | No | No Streamlit or Power BI dashboard is currently included. |
| Business Impact | Is the financial problem clearly articulated? | Yes | The project explains the BNPL credit-risk problem and the need for alternative credit scoring. |
| Business Impact | Are success metrics defined? | Partial | Technical metrics are listed, but verified model results and decision-oriented business metrics are not yet documented. |

## Original Credit-Risk Rubric Gap Analysis

| Original Requirement | Status | Required Improvement |
|---|---|---|
| Credit-scoring business understanding | Partial | Strengthen the connection between Basel II, interpretability, monitoring, and model-governance decisions. |
| Complete EDA | Partial | Run the notebook using the real dataset and document distributions, missing values, outliers, correlations, and 3–5 final insights. |
| Aggregate transaction features | Partial | Existing aggregate features should be verified and expanded with relevant temporal features. |
| Transaction hour, day, month, and year | Missing | Add and test time-based behavioural features where they provide customer-level value. |
| Missing-value handling | Implemented | Verify behaviour with tests and document the selected strategies. |
| Categorical encoding | Implemented | Verify handling of unseen categories during inference. |
| Numerical scaling | Implemented | Ensure the fitted transformation is preserved in the saved model pipeline. |
| Weight of Evidence and Information Value | Missing | Add WoE/IV analysis or provide a justified alternative implementation. |
| RFM proxy-target creation | Partial | Improve high-risk cluster selection and explicitly account for high recency, low frequency, and low monetary value. |
| Deterministic K-Means | Implemented | `random_state` is set; additional validation and cluster-profile reporting are needed. |
| Training/testing split | Implemented in code | Verify it with real data and prevent preprocessing leakage. |
| At least two models | Implemented in code | Run both models and provide real comparison evidence. |
| Hyperparameter tuning | Implemented in code | Run the search and record best parameters. |
| Required evaluation metrics | Implemented in code | Record verified accuracy, precision, recall, F1, and ROC-AUC results. |
| MLflow experiment tracking | Partial | Produce experiment runs, screenshots, artifacts, and registered best-model evidence. |
| Best model registration | Missing | Register and version the selected model in MLflow. |
| Risk-probability output | Partial | Validate the complete raw-input-to-probability path. |
| Credit-score output | Missing | Add a transparent probability-to-score or probability-to-risk-band transformation. |
| Loan amount and duration recommendation | Missing | Add a clearly labelled policy-based recommendation or scenario tool. |
| FastAPI prediction service | Partial | Repair model loading, strengthen validation, and add API integration tests. |
| Docker deployment | Partial | Build and run the image, test the endpoint, and capture evidence. |
| GitHub Actions CI/CD | Partial | Verify a green remote workflow and add a CI badge to the README. |
| Final report | Missing | Create a professional finance-sector technical report or blog post. |
| Required screenshots | Missing | Capture MLflow, CI, Docker, API, dashboard, SHAP, and test evidence. |

## Critical Reliability Risks Identified

### 1. Saved pipeline correctness

The current training workflow trains estimators using already transformed features and later constructs a new feature-engineering pipeline for saving. The newly constructed preprocessing steps are not fitted before serialization. This may cause the saved model to fail when receiving raw transaction data.

### 2. Circular proxy-target risk

Recency, Frequency, and Monetary values define the proxy target and are also included among the model predictors. This can make performance appear stronger because the model is learning the rule used to create the label. The project must either separate proxy-defining variables from prediction variables or clearly measure and disclose the circularity risk.

### 3. Incomplete high-risk cluster definition

The current implementation primarily uses frequency and monetary value to identify the high-risk cluster. The methodology should explicitly include high recency, low frequency, and low monetary value.

### 4. Training execution and model registry

The documented training command does not yet provide a fully verified command-line training workflow. No evidence currently demonstrates successful model training, MLflow comparison, model registration, or artifact loading.

### 5. API reliability

The API does not yet provide sufficient health checks, model-readiness reporting, customer consistency validation, or integration tests. Empty or mixed-customer requests could produce unclear behaviour.

### 6. Lack of decision-oriented explainability

The project does not yet provide global feature importance, individual prediction explanations, risk drivers, model calibration evidence, or threshold analysis.

## Prioritized Week 12 Improvements

### Priority 1: Repair and refactor the end-to-end ML pipeline

**Estimated time: 6–8 hours**

- Make preprocessing, training, serialization, loading, and prediction consistent.
- Add configuration objects using dataclasses.
- Add complete type hints and docstrings.
- Define named constants instead of scattered magic numbers.
- Add a working command-line training entry point.
- Prevent preprocessing leakage.
- Validate a serialized-model round trip.

**Reason:** A reliable saved model is the foundation for every other feature.

### Priority 2: Strengthen testing, API validation, and CI/CD

**Estimated time: 5–6 hours**

- Expand unit and integration tests.
- Test invalid, empty, and mixed-customer requests.
- Test model save-and-load behaviour.
- Test the FastAPI endpoint.
- Add a health/readiness endpoint.
- Run CI on pushes and pull requests.
- Add a CI status badge to the README.

**Reason:** Finance-sector employers need evidence that failures are detected automatically.

### Priority 3: Produce verified model results and explainability

**Estimated time: 8–10 hours**

- Complete the real EDA.
- Improve and document the RFM cluster selection.
- Train and compare at least two models.
- Track experiments in MLflow.
- Register the selected model.
- Add SHAP global and individual explanations.
- Add threshold, error-cost, and calibration analysis.

**Reason:** Model performance must be reproducible, interpretable, and defensible.

### Priority 4: Build an interactive finance decision-support dashboard

**Estimated time: 6–8 hours**

- Create a Streamlit dashboard.
- Display portfolio-level risk metrics.
- Allow customer transaction input or sample selection.
- Present probability, risk band, and score.
- Display individual SHAP explanations.
- Add a clearly labelled loan-policy scenario.
- Communicate proxy-model limitations prominently.

**Reason:** The dashboard makes the project's business value understandable to non-technical stakeholders.

### Priority 5: Complete professional documentation and presentation

**Estimated time: 7–9 hours**

- Rewrite the README around business problem, solution, results, setup, architecture, demo, limitations, and future work.
- Add verified screenshots.
- Produce a professional PDF report or technical blog post.
- Create a finance-sector presentation.
- Prepare a concise oral walkthrough.
- Ensure all numerical claims are supported by evidence.

**Reason:** Strong documentation converts technical work into portfolio and employment impact.

## Proposed Day-by-Day Plan

| Day | Planned Work |
|---|---|
| Day 1 | Complete gap analysis, establish the baseline, add tests, and document risks. |
| Day 2 | Repair the training and serialization pipeline; add configuration, type hints, docstrings, and round-trip tests. |
| Day 3 | Complete EDA, improve feature engineering, add WoE/IV analysis, and strengthen proxy-target methodology. |
| Day 4 | Train and compare models, track experiments in MLflow, analyze thresholds, and register the best model. |
| Day 5 | Strengthen FastAPI, Docker, integration tests, and GitHub Actions. |
| Day 6 | Build the Streamlit dashboard and add SHAP explanations and business-policy scenarios. |
| Day 7 | Complete README, report, screenshots, presentation, final verification, and submission evidence. |

## Target Technical Outcomes

The completed project should demonstrate:

- A deterministic and reproducible pipeline.
- A minimum of five meaningful tests, with a higher target of at least ten unit and integration tests.
- A green GitHub Actions workflow.
- Successful linting with flake8.
- A verified serialized-model round trip.
- Real model-comparison results.
- MLflow experiment and registry evidence.
- A working FastAPI service.
- A successfully built Docker container.
- A functional interactive dashboard.
- Global and individual SHAP explanations.
- Documented model limitations and governance controls.

## Evaluation Metrics

The final model evaluation will include:

- Accuracy.
- Precision.
- Recall.
- F1 score.
- ROC-AUC.
- Precision-Recall AUC.
- Confusion matrix.
- False-negative rate.
- Calibration or Brier score.
- Threshold-based decision analysis.

Because the target is a behavioural proxy, these metrics evaluate consistency with the proxy definition and must not be represented as validated real-world default performance.

## Business-Impact Measurement

Business impact will be communicated through transparent scenarios rather than fabricated savings claims. The dashboard may demonstrate:

- Number and percentage of customers in each risk band.
- False-negative and false-positive trade-offs.
- Approval-rate changes at different thresholds.
- Scenario-based expected loss using explicitly stated assumptions.
- Suggested credit limits and loan durations based on documented policy rules.

All assumptions will be labelled, and no unsupported monetary benefit will be claimed.

## Required Evidence for Final Submission

The final submission will include:

- GitHub repository link.
- Passing local pytest output.
- Green GitHub Actions screenshot.
- MLflow experiment-comparison screenshot.
- MLflow registered-model screenshot.
- Model metric tables and plots.
- API documentation and sample request/response.
- Docker build and running-service screenshot.
- Streamlit dashboard screenshots or deployment link.
- SHAP global explanation.
- SHAP individual explanation.
- Updated README.
- Professional PDF report or published blog post.
- Professional finance-sector presentation.

## Current Baseline

At the beginning of the Week 12 improvement:

- The repository was cloned successfully.
- Work was isolated on `week12-production-upgrade`.
- Dependencies installed successfully in a virtual environment.
- The original two tests passed.
- Four prediction tests were added.
- The complete local test suite now reports six passing tests.
- Local development and generated artifacts were added to `.gitignore`.
- The testing and `.gitignore` improvements were committed separately.

This baseline will be used to compare the original project with the completed Week 12 capstone.