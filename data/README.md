## Data location

This repo expects **you** to provide the raw datasets locally (they are gitignored).

### Put your raw files here

- `data/raw/Fraud_Data.csv`
- `data/raw/IpAddress_to_Country.csv`
- `data/raw/creditcard.csv`

### Outputs

Running Task 1 preprocessing will write artifacts to:

- `data/processed/`

### Task 1 command

```bash
python -m scripts.task1_preprocess --dataset all
```


