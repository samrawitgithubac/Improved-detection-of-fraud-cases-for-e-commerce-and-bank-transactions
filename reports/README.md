## Reports (generated summaries)

This folder is committed and contains **lightweight** outputs (JSON summaries) produced by scripts.

Task 2 generates:

- `reports/task2_fraud_results.json`
- `reports/task2_creditcard_results.json`

These include:

- CV mean/std for **AUC-PR** and **F1**
- Test-set **AUC-PR**, **F1**, and **confusion matrix**

Run:

```bash
python -m scripts.task2_train --dataset all
```


