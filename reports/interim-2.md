## Interim-2 Report — Task 2 (Model Building & Training)

### Goal
Build and evaluate fraud detection models for both datasets using metrics appropriate for class imbalance:

- **AUC-PR** (Average Precision)
- **F1-score**
- **Confusion Matrix**

### Data preparation
- Stratified train/test split (preserves fraud rate)
- Preprocessing applied inside the modeling pipeline
- **SMOTE applied only on training folds** (via `imblearn.Pipeline`)

### Models trained
For each dataset:

1) **Baseline (interpretable)**: Logistic Regression (`class_weight="balanced"`)
2) **Ensemble**: Random Forest (`class_weight="balanced_subsample"`)

### Cross-validation
- **Stratified K-Fold** with **k=5**
- Report **mean ± std** for AUC-PR and F1 on training folds

### Reproducibility
Run Task 2:

```bash
python -m scripts.task2_train --dataset all
```

Artifacts:
- `models/task2_<dataset>_<model>.joblib` (full pipeline)
- `reports/task2_<dataset>_results.json` (metrics summary)

### Results summary
After running Task 2, consult:

- `reports/task2_fraud_results.json`
- `reports/task2_creditcard_results.json`

These files include:
- CV mean/std (AUC-PR, F1)
- Test-set AUC-PR, F1, confusion matrix

### Model selection (business justification)
For fraud detection, model choice balances:

- **Higher AUC-PR** (better ranking of rare fraud cases)
- **F1** trade-off (precision/recall balance)
- Interpretability needs (baseline logistic regression remains valuable for explanation and debugging)

In practice, the ensemble often improves recall/precision on rare positives, but the baseline provides a transparent reference for stakeholders.


