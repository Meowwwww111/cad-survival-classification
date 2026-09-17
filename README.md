# Coronary artery disease: survival classification

Python 3.11 submission comparing **logistic regression, random forest and gradient boosting**, with a dummy baseline, patient-group cross-validation and a held-out test set.

Read the [evaluation report](reports/REPORT.md) for the selected model, measured performance, comparison chart and limitations. Aggregate results are committed; patient data and fitted models remain local.

## Run in VS Code or a terminal

Install Python **3.11**, open this folder in VS Code and select the `.venv` interpreter. Place the supplied CSV at `data/classification_bonus_practice_data.csv`.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python survival.py train
.venv\Scripts\python -m pytest -q
```

If using uv (also installs Python if needed):

```powershell
uv venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv\Scripts\python survival.py train
```

On macOS/Linux use `python3.11 -m venv .venv` and `.venv/bin/python` instead. A complete environment snapshot is in `requirements-lock.txt`; `requirements.txt` pins the direct dependencies.

Use an alternate data location:

```powershell
.venv\Scripts\python survival.py train --data "C:/path/classification_bonus_practice_data.csv"
```

Predict on a new CSV containing the same feature columns (Survive is not required):

```powershell
.venv\Scripts\python survival.py predict --data new_patients.csv --output models/predictions.csv
```

`survival_probability` estimates the supplied label, with an unspecified follow-up horizon. `risk_flag=1` indicates estimated non-survival probability at or above the training-selected threshold. Only load trusted joblib files.

## Method

- Assume 1/Yes denotes survival; 0/No denotes non-survival. Internally non-survival is positive to make recall reflect risk detection.
- Normalize categories, treat invalid ages as missing, impute within each fold, encode categories, and scale numeric predictors inside a pipeline.
- Exclude ID and Favorite color from predictors. Keep shared IDs and exact cleaned predictor duplicates in the same split, including connected groups.
- Reserve about 20% by grouped stratified splitting. Use identical five-fold grouped cross-validation for the remaining rows.
- Select the classifier with the highest training CV average precision. Choose an illustrative risk threshold by training out-of-fold F2. Test data influence neither choice.
- Compare ROC AUC, average precision, accuracy, balanced accuracy, precision, recall, F2, Brier score and confusion counts. Include paired group-bootstrap AUC intervals.

Model settings are fixed in `candidates()` so the comparison is transparent. This exercise does not claim exhaustive optimization. Grouped splitting and fold-local preprocessing follow [scikit-learn's leakage guidance](https://scikit-learn.org/1.6/common_pitfalls.html#data-leakage).

## Files

| File | Purpose |
| --- | --- |
| `survival.py` | Cleaning, grouping, training, selection, evaluation and prediction CLI |
| `tests/test_survival.py` | Outcome mapping, invalid-input handling and leakage safeguards |
| `reports/REPORT.md` | Written evaluation and model suitability discussion |
| `reports/cross_validation.csv` | Training CV means and fold standard deviations |
| `reports/test_metrics.csv` | Comparable held-out metrics at threshold 0.50 |
| `reports/results.json` | Selected threshold, metrics, uncertainty intervals and versions |
| `reports/data_audit.json` | Data-quality counts and original CSV checksum |
| `reports/comparison.png` | ROC and precision-recall curves |
| `models/best_model.joblib` | Locally generated trained pipeline; excluded from Git |

## Scope

This educational analysis has no documented follow-up duration, label dictionary, censoring information or external validation. Repeated IDs with conflicting outcomes remain a material data-quality limitation. The model predicts recorded outcomes and does not estimate treatment effects or establish a safe clinical decision rule. Dataset provenance and clinical suitability need independent confirmation.

The repository is private by default. Grant your evaluator access through GitHub repository settings when submitting its URL.
