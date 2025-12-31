## Final Report — End-to-End Fraud Detection (E-commerce + Bank)

### 1) Executive summary
This project built an end-to-end fraud detection workflow for Adey Innovations Inc. covering:

- **E-commerce fraud** with **IP→country geolocation** enrichment and behavior/time features
- **Bank card fraud** with anonymized PCA features

The solution prioritizes imbalanced-learning best practices: stratified splits, train-only resampling, and AUC‑PR/F1 evaluation.

### 2) Data understanding and business context
Fraud detection is highly imbalanced. Operationally:

- False positives reduce customer trust and increase support cost
- False negatives directly increase fraud losses

Therefore, we evaluate models beyond accuracy using **AUC‑PR**, **F1**, and confusion matrices.

### 3) Task 1 — Data preprocessing & feature engineering

#### Cleaning
- Removed duplicates
- Fixed data types (timestamps/numerics)
- Managed missing values with practical imputations and safe drops for critical fields

#### Geolocation integration (Fraud_Data)
- Converted IP to integer
- Range-joined into `IpAddress_to_Country.csv`
- Assigned missing/invalid IPs to **Unknown**

#### Features (Fraud_Data)
- Time: `hour_of_day`, `day_of_week`, `time_since_signup_sec`
- Velocity: `user_txn_count_1h`, `user_txn_count_24h`
- Aggregates: user/device counts

#### Transformations
- Scaling: `StandardScaler`
- Encoding: `OneHotEncoder`
- Imbalance handling: **SMOTE on training data only**

### 4) Task 2 — Modeling & evaluation
Models:

- Baseline: Logistic Regression (interpretable)
- Ensemble: Random Forest

Validation:

- Stratified K-Fold CV (k=5)
- Test evaluation with AUC‑PR / F1 / confusion matrix

Outputs:

- Metrics: `reports/task2_*_results.json`
- Models: `models/task2_*_*.joblib`

### 5) Task 3 — Explainability (SHAP)

#### 5.1 Feature Importance Baseline
- Extracted built-in feature importance from ensemble models
- Visualized top 10 most important features for both datasets
- Provides baseline understanding of model's feature usage

#### 5.2 SHAP Analysis

**Global Explanations:**
- SHAP Summary Plot (beeswarm) showing global feature importance
- Identifies features that consistently drive fraud predictions

**Local Explanations:**
- SHAP Waterfall plots for individual predictions:
  - **True Positive (TP)**: Correctly identified fraud case
  - **False Positive (FP)**: Legitimate transaction incorrectly flagged
  - **False Negative (FN)**: Missed fraud case
- Provides insight into why specific predictions were made

#### 5.3 Interpretation

**Comparison:**
- Compared SHAP importance with built-in feature importance
- Identified top 5 drivers of fraud predictions
- Found discrepancies indicating complex feature interactions

**Key Findings:**
- Time-based features (time_since_signup, hour_of_day) are strong fraud indicators
- Transaction velocity features show high importance
- Device/user aggregation features reveal behavioral patterns
- Country features enable geographic risk assessment

#### 5.4 Business Recommendations

Based on SHAP analysis, actionable recommendations include:

1. **Transaction Timing & Signup Window**
   - Transactions within 24-48 hours of signup should receive additional verification (OTP/2FA)
   - SHAP Evidence: Time-based features show significant impact

2. **Transaction Velocity Monitoring**
   - Implement real-time velocity checks:
     - Flag users with >3 transactions in 1 hour for manual review
     - Block users with >10 transactions in 24 hours until verified
   - SHAP Evidence: Velocity features are key fraud indicators

3. **Device & User Behavior Patterns**
   - Monitor device-user relationships:
     - Flag devices associated with >5 unique users in 30 days
     - Require verification for users switching devices frequently
   - SHAP Evidence: Device/user features reveal fraud patterns

4. **Geographic Risk Assessment**
   - Implement country-based risk scoring:
     - High-risk countries: Require additional verification
     - Mismatch between IP country and billing address: Flag for review
   - SHAP Evidence: Country features show varying fraud risk levels

5. **Transaction Value Thresholds**
   - Implement tiered verification based on transaction value:
     - Low value (<$50): Standard processing
     - Medium value ($50-$500): Additional verification if combined with other risk factors
     - High value (>$500): Always require step-up authentication
   - SHAP Evidence: Value features impact fraud probability

**Implementation:**
- Module: `src/modeling/task3_shap.py`
- Notebook: `notebooks/shap-explainability.ipynb`
- Dependencies: `requirements-task3.txt` (shap==0.45.1)

These recommendations should be tested in a controlled environment and adjusted based on business constraints and false positive tolerance.

### 7) How to reproduce

Install base deps:

```bash
pip install -r requirements.txt
```

Task 1:

```bash
python -m scripts.task1_preprocess --dataset all
```

Task 2:

```bash
python -m scripts.task2_train --dataset all
```

Task 3:

```bash
pip install -r requirements-task3.txt
```

Then open:
- `notebooks/shap-explainability.ipynb`

The notebook provides:
- Feature importance baseline visualization
- SHAP summary plots (global feature importance)
- SHAP waterfall plots for TP, FP, FN cases
- Comparison between SHAP and built-in importance
- Top 5 drivers of fraud predictions
- Business recommendations based on SHAP insights


