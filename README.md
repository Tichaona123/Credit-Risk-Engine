# Enterprise Credit Risk IFRS9 Analytics

This repository contains the `Enterprise_Credit_Risk_IFRS9_Analytics.ipynb` notebook, which presents an enterprise-grade credit risk analytics solution for loan default prediction and IFRS 9 expected credit loss modeling.

## Project Overview

The notebook demonstrates a complete analytical workflow for credit risk modeling, including:
- Business and regulatory context for credit risk and IFRS 9
- Data loading, profiling, and quality assessment
- Feature engineering and domain-driven transformation
- Adversarial validation for train/test distribution assessment
- Training of CatBoost, LightGBM, and XGBoost models
- Ensemble construction, calibration, and performance evaluation
- SHAP explainability analysis for model transparency
- Statistical testing, error analysis, and risk interpretation
- IFRS 9 staging and expected credit loss (ECL) framework
- Model governance, monitoring, and deployment preparation

## Key Notebook Sections

1. Executive Summary
2. Business Understanding and Problem Statement
3. IFRS 9 Regulatory Framework
4. Data Loading, Configuration, and Quality Assessment
5. Feature Engineering
6. Adversarial Validation
7. Model Training and Hyperparameter Optimization
8. Ensemble Construction
9. Comprehensive Model Evaluation
10. SHAP Explainability Analysis
11. Statistical Testing
12. Error Analysis
13. IFRS 9 Staging and ECL Framework
14. Model Governance and Monitoring
15. Stress Testing
16. Deployment and Artifacts
17. Conclusions and Recommendations

## Data Files

The notebook relies on the following dataset files included in this folder:
- `Train.csv`
- `Test.csv`
- `VariableDefinitions.csv`
- `SampleSubmission.csv`

Additional training artifacts and model output files are stored under:
- `catboost_info/`

## Usage

To explore the notebook:
1. Open `credit_risk_project/Enterprise_Credit_Risk_IFRS9_Analytics.ipynb` in Jupyter Notebook or JupyterLab.
2. Execute the notebook cells in order to reproduce the analysis and results.

> Note: The notebook includes imports and environment configuration sections. Ensure the required Python packages are installed before running the notebook.

## Purpose

This notebook is designed for:
- credit risk practitioners building IFRS 9-compliant PD/ECL analytics
- data scientists demonstrating explainable machine learning in banking
- risk model governance teams reviewing model development documentation
- analysts evaluating the predictive performance of ensemble gradient boosting models

## Output and Deliverables

The notebook produces:
- trained model predictions and validation metrics
- SHAP explainability visualizations
- staging and ECL analysis supporting IFRS 9 requirements
- deployment-ready artifacts and governance documentation

## Notes

The analysis emphasizes regulatory compliance, model robustness, and explainability. It is intended as a demonstrative enterprise solution rather than a production-ready application without further validation and operational integration.
