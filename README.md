# Customer Churn Prediction

A telecom churn classification project using Python, pandas, and scikit-learn. The original exploratory notebooks are retained; `train.py` provides the repeatable training workflow.

## Run locally

Use Python 3.11 or newer. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
```

On macOS/Linux, use `.venv/bin/python` instead. To use a different input or output directory:

```sh
python train.py --data path/to/customers.csv --output artifacts --seed 42
```

The included CSV is `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. It uses the IBM Telco Customer Churn schema: one customer per row, a Yes/No `Churn` target, and account, service, and demographic features. It is a static educational dataset, not live customer data.

## Evaluation design

- Validate target labels and unique customer IDs; exclude IDs from predictors.
- Convert whitespace/non-numeric TotalCharges to missing values.
- Reserve a stratified 20% test split with a fixed seed.
- Fit median imputation, numerical scaling, categorical imputation, and one-hot encoding within each training fold. Unknown categories are supported at prediction time.
- Compare a dummy baseline, logistic regression, balanced logistic regression, and balanced random forest with five-fold stratified cross-validation.
- Select by mean cross-validation average precision, then evaluate the winner on the held-out test split at a fixed 0.5 threshold.
- Report precision, recall, F1, ROC-AUC, average precision, accuracy, and a confusion matrix. Average precision summarizes precision-recall performance and is useful for this imbalanced target.

## Outputs

`artifacts/metrics.json` records model comparisons, held-out metrics, seed, split sizes, and library versions. `artifacts/churn_pipeline.joblib` saves preprocessing and the selected classifier together. Outputs are ignored by Git. The validated run is recorded in `results/metrics.json`; rerunning writes fresh results under `artifacts/`.

## Exploration

Install `requirements-notebook.txt` for Jupyter and matplotlib. Run notebooks from the repository root. The two notebooks are historical exploratory variants, not the authoritative evaluation workflow: their manual preprocessing differs from `train.py`. Use the script for model comparison and reported results.

## Limitations and next steps

This is an offline educational model, not a production deployment. Random splitting does not establish performance on future customers. Correlations and feature importance do not prove causal retention effects. No computer vision, PyTorch, or edge deployment is implemented.

Useful extensions are validation-only threshold selection based on outreach costs, hyperparameter searches confined to training folds, calibration analysis, and a batch prediction interface. Keep test data separate from these decisions.

Only load joblib artifacts you trust. To preserve an exact tested environment, save `python -m pip freeze` to a lock file after installation.

## Validated baseline

On the included 7,043-row CSV, seed 42, with 5,634 training rows and 1,409 held-out rows, logistic regression was selected by training-fold average precision.

| Held-out metric | Value |
| --- | ---: |
| ROC-AUC | 0.8419 |
| Average precision | 0.6334 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| F1 | 0.6040 |
| Accuracy | 0.8055 |

These are baseline results, not evidence of improvement over the historical notebook. The fixed threshold is 0.5. Balanced logistic regression had higher training cross-validation recall (0.8013 versus 0.5431), illustrating the need to select metrics and thresholds around business costs.

The tested environment is captured in `requirements-lock.txt`; install it with `python -m pip install -r requirements-lock.txt`. Run the raw-input, unseen-category, and duplicate-ID check with `python -m unittest test_train.py`.
