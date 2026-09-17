"""Dataset validation, fixed cleaning rules and fold-local preprocessing."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["Age", "Sodium", "Creatinine", "Platelets", "Creatine phosphokinase",
           "Blood Pressure", "Hemoglobin", "Height", "Weight"]
CATEGORICAL = ["Gender", "Smoke", "Diabetes", "Ejection Fraction"]
FEATURES = NUMERIC + CATEGORICAL


class CleanFeatures(TransformerMixin, BaseEstimator):
    """Fixed rules only; learned imputation happens later inside each fold."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing = set(FEATURES) - set(X.columns)
        if missing:
            raise ValueError(f"Missing feature columns: {sorted(missing)}")
        out = X[FEATURES].copy()
        for col in NUMERIC:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[~np.isfinite(out[col]) | (out[col] <= 0), col] = np.nan
        # Negative ages are invalid, not assumed to be sign errors.
        out.loc[out.Age > 120, "Age"] = np.nan
        for col in CATEGORICAL:
            out[col] = out[col].astype("string").str.strip().str.lower()
        out["Ejection Fraction"] = out["Ejection Fraction"].replace({"l": "low", "n": "normal"})
        out["Smoke"] = out["Smoke"].replace({"y": "yes", "n": "no"})
        return out.replace({pd.NA: np.nan})


def load_data(path):
    raw = pd.read_csv(path)
    required = set(FEATURES + ["ID", "Survive"])
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(raw.columns))}")
    target = raw.Survive.astype("string").str.strip().str.lower().map(
        {"0": 0, "1": 1, "no": 0, "yes": 1})
    if target.isna().any():
        raise ValueError("Missing or unrecognized Survive label; refusing to guess.")
    if raw.ID.isna().any():
        raise ValueError("Missing patient ID; cannot establish patient separation.")
    y = (1 - target).astype(int).to_numpy()
    # Union connected records sharing ID OR exactly matching cleaned predictors.
    # This uses no target information and prevents exact-copy leakage across IDs.
    parent = np.arange(len(raw))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    clean = CleanFeatures().fit_transform(raw)
    for keys in (raw.ID.astype(str).str.strip(),
                 pd.util.hash_pandas_object(clean, index=False)):
        seen = {}
        for i, key in enumerate(keys):
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = np.array([root(i) for i in range(len(raw))])
    audit = {
        "rows": len(raw), "unique_patient_ids": int(raw.ID.nunique()),
        "independent_groups": int(len(np.unique(groups))),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_cleaned_predictors": int(clean.duplicated().sum()),
        "ids_with_conflicting_outcomes": int(pd.DataFrame({"id": raw.ID, "y": y}).groupby("id").y.nunique().gt(1).sum()),
        "survival_count": int((y == 0).sum()), "non_survival_count": int(y.sum()),
        "negative_age_count": int((pd.to_numeric(raw.Age, errors="coerce") < 0).sum()),
        "missing_after_cleaning": clean.isna().sum().to_dict(),
        "raw_categories": {c: sorted(raw[c].dropna().astype(str).unique().tolist()) for c in CATEGORICAL},
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    return raw, y, groups, audit


def make_pipeline(model, numerical=None, nominal=None):
    numerical = NUMERIC if numerical is None else numerical
    nominal = CATEGORICAL if nominal is None else nominal
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return Pipeline([("clean", CleanFeatures()), ("prepare", ColumnTransformer(
        [("numeric", numeric, numerical), ("categorical", categorical, nominal)])),
        ("model", model)])


class DataPreparation:
    """Prepare this dataset using the configured predictor subsets.

    All learned transformations stay inside the estimator pipeline. Cleaning
    and patient grouping happen before splitting using fixed, target-free rules.
    """
    def __init__(self, config):
        self.config = config
        numerical = config["numerical_features"]
        nominal = config["nominal_features"]
        if not set(numerical).issubset(NUMERIC) or not set(nominal).issubset(CATEGORICAL):
            raise ValueError("Configured predictors must use the supported feature groups")
        if not numerical and not nominal:
            raise ValueError("At least one predictor is required")
        if len(numerical + nominal) != len(set(numerical + nominal)):
            raise ValueError("Configured predictors must be unique")

    def load_data(self, path):
        return load_data(path)

    def clean_data(self, frame):
        return CleanFeatures().fit_transform(frame)

    def make_pipeline(self, model):
        return make_pipeline(model, self.config["numerical_features"], self.config["nominal_features"])
