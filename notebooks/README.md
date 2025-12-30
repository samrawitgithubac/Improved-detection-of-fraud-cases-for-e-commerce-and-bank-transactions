## Notebooks

These notebooks are organized to match the project tasks.

### Task 1

- `eda-fraud-data.ipynb`: EDA for `Fraud_Data.csv` (+ country analysis after IP merge)
- `eda-creditcard.ipynb`: EDA for `creditcard.csv`
- `feature-engineering.ipynb`: Feature engineering + preprocessing (calls the same code as the scripts)

### Task 2

- `modeling.ipynb`: baseline + ensemble models with AUC-PR / F1 / confusion matrix and Stratified K-Fold CV

### Task 3

- `shap-explainability.ipynb`: SHAP global + local explanations (TP / FP / FN)

### Data paths

Raw data goes in `data/raw/`:

- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

Processed outputs go to `data/processed/` (gitignored).


