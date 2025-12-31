## Interim-3 Report — Task 3 (Model Explainability)

### Goal
Interpret the best Task 2 model's predictions using SHAP to understand what drives fraud detection and provide actionable business recommendations.

### Approach

#### 1. Feature Importance Baseline
- Extract built-in feature importance from the ensemble model (Random Forest `feature_importances_` or Logistic Regression `coef_`)
- Visualize top 10 most important features
- Provides baseline understanding of model's feature usage

#### 2. SHAP Analysis

**Global Explanations:**
- Generate SHAP Summary Plot (beeswarm) showing global feature importance
- Identifies which features most consistently drive fraud predictions across all samples

**Local Explanations:**
- Generate SHAP Force/Waterfall Plots for at least 3 individual predictions:
  - **True Positive (TP)**: Correctly identified fraud case
  - **False Positive (FP)**: Legitimate transaction incorrectly flagged as fraud
  - **False Negative (FN)**: Missed fraud case
- Provides insight into why specific predictions were made

#### 3. Interpretation

**Comparison:**
- Compare SHAP importance with built-in feature importance
- Identify discrepancies that may indicate:
  - Complex feature interactions (high SHAP, low built-in)
  - Features with less direct impact (low SHAP, high built-in)

**Top 5 Drivers:**
- Identify the top 5 features driving fraud predictions based on SHAP analysis
- Explain any surprising or counterintuitive findings

#### 4. Business Recommendations

Provide at least 3 actionable recommendations based on SHAP insights, such as:

1. **Transaction Timing & Signup Window**
   - Recommendation: Transactions within 24-48 hours of signup should receive additional verification (OTP/2FA)
   - SHAP Evidence: Time-based features show significant impact on fraud probability

2. **Transaction Velocity Monitoring**
   - Recommendation: Implement real-time velocity checks (e.g., flag >3 transactions in 1 hour)
   - SHAP Evidence: Velocity features are key fraud indicators

3. **Device & User Behavior Patterns**
   - Recommendation: Monitor device-user relationships (flag devices with >5 unique users)
   - SHAP Evidence: Device/user aggregation features reveal fraud patterns

4. **Geographic Risk Assessment**
   - Recommendation: Implement country-based risk scoring with additional verification for high-risk countries
   - SHAP Evidence: Country features show varying fraud risk levels

5. **Transaction Value Thresholds**
   - Recommendation: Implement tiered verification based on transaction value
   - SHAP Evidence: Purchase value features impact fraud probability

### Implementation

**Module:** `src/modeling/task3_shap.py`
- `explain_task3()`: Main function that loads best model, computes SHAP values, finds TP/FP/FN examples
- Handles both Random Forest (TreeExplainer) and Logistic Regression (LinearExplainer)
- Extracts built-in feature importance for comparison

**Notebook:** `notebooks/shap-explainability.ipynb`
- Complete workflow for both Fraud_Data and CreditCard datasets
- All visualizations and analysis in one place

### Dependencies

Install Task 3 dependencies:

```bash
pip install -r requirements-task3.txt
```

Requires: `shap==0.45.1`

### Reproducibility

1. Ensure Task 2 is complete (models exist in `models/` directory)

2. Run the notebook:

```bash
jupyter notebook notebooks/shap-explainability.ipynb
```

Or execute cells programmatically.

### Key Insights

**Fraud_Data Dataset:**
- Time-based features (time_since_signup, hour_of_day) are strong fraud indicators
- Transaction velocity features (user_txn_count_1h, user_txn_count_24h) show high importance
- Device and user aggregation features reveal behavioral patterns
- Country features enable geographic risk assessment

**CreditCard Dataset:**
- PCA features (V1-V28) make direct business interpretation challenging
- Recommendations focus on transaction amount monitoring and anomaly detection
- Time-based patterns and rapid successive transactions are key indicators

### Business Value

SHAP explainability enables:
- **Transparency**: Understand why specific transactions are flagged
- **Trust**: Stakeholders can verify model reasoning
- **Actionability**: Convert model insights into operational rules
- **Debugging**: Identify and fix model biases or errors
- **Compliance**: Meet regulatory requirements for explainable AI

### Next Steps

1. Validate recommendations in a controlled environment
2. Adjust thresholds based on business constraints and false positive tolerance
3. Implement real-time monitoring based on SHAP insights
4. Continuously refine rules as fraud patterns evolve

