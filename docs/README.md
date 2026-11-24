# APEX Architecture & Documentation

## Overview
APEX is an Intelligent Predictive Maintenance and MLOps Platform designed for industrial equipment monitoring, risk forecasting, anomaly detection, and automated model governance.

## System Components
1. **Core API (`backend/app/main.py`)**:
   - High-throughput FastAPI service.
   - Authentication (OAuth2 Bearer with JWT access tokens and revocable refresh tokens).
   - Sensor validation and sanitization.
   - Batch CSV inference processing.
   - Unsupervised sensor anomaly detection.
   - Tree-based SHAP feature attribution.
   - Production input drift detection (Population Stability Index + Kolmogorov-Smirnov).

2. **Machine Learning Pipeline (`ml/`)**:
   - `ml/data/ai4i.py`: AI4I data simulation and real UCI loader.
   - `ml/features/engineering.py`: Stateless feature engineering (temperature difference, mechanical power, mechanical strain, wear rate).
   - `ml/training/train.py`: Multi-model training and selection (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM) optimized for validation PR-AUC and F2-score thresholding.
   - `ml/anomaly_detection/detector.py`: Unsupervised outlier detectors (Isolation Forest, Local Outlier Factor, One-Class SVM).
   - `ml/monitoring/drift.py`: Statistical drift reporting across numeric and categorical features.

3. **Model Governance & Registry (`backend/app/services/registry.py`)**:
   - Strict candidate lifecycle (Candidate -> Active -> Archived / Rolled Back).
   - Enforced promotion policies preventing recall or PR-AUC regressions without explicit admin override.
   - Model artifact integrity verification via SHA-256 checksums.

---
## Author
**Aastha Chauhan**  
GitHub: [@Aasthachauhan](https://github.com/Aasthachauhan)
