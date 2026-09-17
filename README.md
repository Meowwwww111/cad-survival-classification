# ❤️ Coronary Artery Disease Survival Predictor

A Python 3.11 machine learning project for predicting the recorded survival of coronary artery disease patients. The project covers data cleaning, preprocessing, model comparison, patient-group cross-validation and final test-set evaluation.

Organized like [AIAP_HDB_Price_Predictor](https://github.com/Meowwwww111/AIAP_HDB_Price_Predictor), with `main.py`, modular classes under `src/`, a YAML configuration and an exploratory notebook.

## 📌 Project Overview

The workflow is:

1. Load and validate the supplied patient dataset.
2. Normalize labels and categories, and identify invalid or missing values.
3. Group shared patient IDs and exact copies of cleaned predictors.
4. Reserve approximately 20% of rows as a grouped, stratified test set.
5. Fit preprocessing separately within each training fold.
6. Compare three classifiers and a dummy baseline using five-fold grouped cross-validation.
7. Select the model with the highest mean validation average precision.
8. Choose a risk threshold using training out-of-fold F2.
9. Evaluate the models on the held-out test set and save the report and trained pipeline.

## 🤖 Models

- Logistic Regression
- Random Forest
- Histogram Gradient Boosting
- Dummy prior-probability baseline

Model parameters are declared in `src/config.yaml`. The exercise uses fixed configurations rather than a hyperparameter search. Random forest is the nominal winner in the default experiment; gradient boosting is practically tied. See the [full evaluation report](reports/REPORT.md).

## 🧹 Data Preparation

- Map `Survive=1/Yes` to survival and `0/No` to non-survival.
- Use non-survival as the internal positive class so recall measures risk detection.
- Normalize categorical whitespace, case and abbreviations.
- Treat negative ages, ages above 120 and nonpositive numerical measurements as missing.
- Fit median numerical and mode categorical imputation inside each training fold.
- Standardize numerical features and one-hot encode categories.
- Exclude ID and Favorite color from model predictors.
- Keep repeated IDs and exact predictor copies together across every split.

Conflicting outcomes for repeated IDs are retained and reported because the CSV provides no timestamps or explanation of those records.

## 📊 Features

### Numerical

Age, Sodium, Creatinine, Platelets, Creatine phosphokinase, Blood Pressure, Hemoglobin, Height and Weight.

### Nominal categorical

Gender, Smoke, Diabetes and Ejection Fraction.

### Target

`Survive`. The outcome meaning is assumed from the column name and requires confirmation from the data owner. No follow-up horizon or censoring information is provided.

## 📈 Evaluation

Models are compared using average precision, ROC AUC, accuracy, balanced accuracy, risk precision, risk recall, survival recall, F2, Brier score and confusion counts. Patient-group bootstrap intervals assess held-out AUC uncertainty.

Model selection uses training cross-validation only. The held-out test set does not select models, parameters or the operating threshold. Fold-local preprocessing follows [scikit-learn's leakage guidance](https://scikit-learn.org/1.6/common_pitfalls.html#data-leakage).

## 📁 Project Structure

```text
cad-survival-classification/
├── data/
│   ├── README.md
│   └── classification_bonus_practice_data.csv  # local, excluded from Git
├── src/
│   ├── __init__.py
│   ├── config.yaml
│   ├── data_preparation.py
│   └── model_training.py
├── eda.ipynb
├── main.py
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
├── .python-version
├── README.md
├── reports/          # evaluation report, metrics and chart
├── tests/            # data and configuration checks
├── .github/workflows/tests.yml
├── models/           # generated locally, excluded from Git
└── survival.py       # compatibility entry point for earlier commands
```

## 📄 File Description

| File | Description |
| --- | --- |
| `main.py` | Main entry point for configuration, data loading, training and prediction |
| `src/data_preparation.py` | `DataPreparation` class, cleaning transformer, patient grouping and preprocessing |
| `src/model_training.py` | `ModelTraining` class, model definitions, comparison, selection and evaluation |
| `src/config.yaml` | Dataset/output paths, feature subsets, split settings and model parameters |
| `eda.ipynb` | Exploratory analysis of data quality and training-set feature distributions |
| `data/classification_bonus_practice_data.csv` | Supplied patient dataset, kept locally |
| `reports/REPORT.md` | Model suitability discussion, measured results and limitations |
| `reports/cross_validation.csv` | Training CV means and fold standard deviations |
| `reports/test_metrics.csv` | Held-out metrics at the common threshold of 0.50 |
| `reports/results.json` | Selected threshold, bootstrap intervals and experiment settings |
| `reports/data_audit.json` | Data-quality counts and source checksum |
| `reports/comparison.png` | ROC and precision-recall curves |
| `models/best_model.joblib` | Locally generated fitted pipeline |
| `requirements.txt` | Pinned direct Python dependencies |

## ⚙️ Configuration

Edit `src/config.yaml` to change the dataset location, report/model directories, random seed, grouped holdout/CV fold counts, worker count, threshold grid, bootstrap repetitions, predictor subsets or classifier parameters.

Relative paths in the configuration resolve from the project root. CLI path overrides resolve from the current directory. Outcome coding and patient-ID handling remain fixed to the supplied dataset. Keep gradient boosting's internal early stopping disabled so it cannot introduce an ungrouped validation split. Changing settings creates a new experiment; the committed report describes the defaults.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Meowwwww111/cad-survival-classification.git
cd cad-survival-classification
```

The repository is private. Sign in with an authorized GitHub account and grant the evaluator access before submitting.

### 2. Create a Python 3.11 environment

Using Conda:

```bash
conda create -n cad-survival python=3.11 -y
conda activate cad-survival
```

Or on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux use `python3.11 -m venv .venv` and `source .venv/bin/activate`. In VS Code select this environment as the Python interpreter and notebook kernel.

### 3. Install dependencies and add the dataset

```bash
python -m pip install -r requirements.txt
```

Place the supplied CSV at `data/classification_bonus_practice_data.csv`. Patient records and fitted models are excluded from Git. `requirements-lock.txt` records the complete environment used for the committed experiment.

### 4. Run the model pipeline

```bash
python main.py
```

This uses `src/config.yaml` and generates aggregate reports plus `models/best_model.joblib`.

Optional overrides:

```bash
python main.py train --data "C:/path/classification_bonus_practice_data.csv" --jobs 2
python main.py train --config src/config.yaml
```

### 5. Predict new records or run tests

```bash
python main.py predict --data new_patients.csv --output models/predictions.csv
python -m pytest -q
```

New records need the same feature columns; `Survive` is not required. Output includes survival probability, non-survival probability and a risk flag at the training-selected threshold. Only load trusted joblib files. Earlier `python survival.py train` and `predict` commands continue to work.

## 🔎 Exploratory Data Analysis

Open `eda.ipynb` in VS Code and select the Python 3.11 environment. The notebook examines schema, missing values, repeated IDs, outcome balance, numerical distributions and correlations, then displays saved model-comparison results.

Outcome relationships use only the training partition. Committed notebook outputs contain aggregate summaries rather than patient-level records.

## 🛠️ Technologies Used

Python 3.11, pandas, NumPy, scikit-learn, PyYAML, Matplotlib, Jupyter and pytest.

## ⚠️ Notes & Limitations

- The near-perfect tree-model scores warrant investigation of synthetic construction, repeated underlying source records and outcome-derived features.
- The dataset has conflicting outcomes for some repeated IDs and unknown provenance.
- No follow-up horizon, data dictionary or censoring information is supplied.
- These classifiers estimate the recorded outcome; they do not estimate treatment effects or survival time.
- External validation, probability calibration, subgroup assessment and clinical utility testing are needed before considering use in patient care.

## 🔮 Potential Improvements

- Confirm outcome coding, measurement units, feature timing and source provenance.
- Investigate conflicting IDs and possible repeated source records.
- Evaluate calibration, subgroup performance and external generalization.
- Add clinical expert review of features and decision thresholds.

## 👨‍💻 Author

**Jasper Ng (Ng Jing Heng)**  
NTU Mechanical Engineering — Intelligent Manufacturing

## 📜 License

Intended for educational and portfolio purposes. No license to redistribute the source patient dataset is assumed.
