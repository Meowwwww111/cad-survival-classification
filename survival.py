"""Reproducible survival classification exercise; Python 3.11.

Internally, positive class = non-survival (risk), assuming Survive 1/Yes means
survival. This convention makes recall measure detection of patients at risk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, fbeta_score, precision_score,
    recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
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


def make_pipeline(model):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return Pipeline([("clean", CleanFeatures()), ("prepare", ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])),
        ("model", model)])


def candidates():
    # Fixed, predeclared configurations. No test-driven hyperparameter search.
    return {
        "Dummy baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
        "Random forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=5,
            max_features="sqrt", n_jobs=1, random_state=SEED),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=200,
            max_leaf_nodes=15, learning_rate=0.05, l2_regularization=1.0,
            early_stopping=False, random_state=SEED),
    }


def metrics(y, probability, threshold=0.5):
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(roc_auc=roc_auc_score(y, probability),
        average_precision=average_precision_score(y, probability),
        accuracy=accuracy_score(y, pred), balanced_accuracy=balanced_accuracy_score(y, pred),
        risk_recall=recall_score(y, pred, zero_division=0),
        risk_precision=precision_score(y, pred, zero_division=0),
        risk_f2=fbeta_score(y, pred, beta=2, zero_division=0),
        survival_recall=tn / (tn + fp), brier=brier_score_loss(y, probability),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def bootstrap_auc(y, probabilities, groups, winner, repetitions=500):
    """Paired cluster bootstrap: same sampled patient groups for every model."""
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    positions = [np.flatnonzero(groups == g) for g in unique]
    samples = {name: [] for name in probabilities}
    for _ in range(repetitions):
        idx = np.concatenate([positions[i] for i in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(y[idx])) < 2:
            continue
        for name, p in probabilities.items():
            samples[name].append(roc_auc_score(y[idx], p[idx]))
    result = {}
    for name, vals in samples.items():
        delta = np.array(samples[winner]) - np.array(vals)
        result[name] = {"auc_95_percentile_interval": np.quantile(vals, [0.025, 0.975]).tolist(),
                        "winner_minus_model_auc_interval": np.quantile(delta, [0.025, 0.975]).tolist()}
    return result


def markdown_table(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw, y, groups, audit = load_data(args.data)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(outer.split(raw, y, groups))
    Xtr, Xte, ytr, yte = raw.iloc[train_idx], raw.iloc[test_idx], y[train_idx], y[test_idx]
    gtr, gte = groups[train_idx], groups[test_idx]
    assert not set(gtr) & set(gte)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1).split(Xtr, ytr, gtr))
    for a, b in folds:
        assert not set(gtr[a]) & set(gtr[b])
    audit.update(train_rows=len(ytr), test_rows=len(yte),
                 train_risk_prevalence=float(ytr.mean()), test_risk_prevalence=float(yte.mean()))
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pipes = {name: make_pipeline(model) for name, model in candidates().items()}
    cv_rows = []
    for name, pipe in pipes.items():
        print(f"Cross-validating {name} ...", flush=True)
        scores = cross_validate(pipe, Xtr, ytr, cv=folds, n_jobs=args.jobs,
            scoring={"auc": "roc_auc", "ap": "average_precision", "brier": "neg_brier_score"}, error_score="raise")
        cv_rows.append(dict(model=name, cv_auc=float(scores["test_auc"].mean()),
            cv_auc_sd=float(scores["test_auc"].std(ddof=1)), cv_ap=float(scores["test_ap"].mean()),
            cv_ap_sd=float(scores["test_ap"].std(ddof=1)), cv_brier=float(-scores["test_brier"].mean())))
    cv = pd.DataFrame(cv_rows).sort_values("cv_ap", ascending=False)
    winner = cv[cv.model != "Dummy baseline"].iloc[0].model
    # Lock winner and threshold using training data only, before looking at test performance.
    oof = cross_val_predict(pipes[winner], Xtr, ytr, cv=folds, method="predict_proba", n_jobs=args.jobs)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    f2 = [fbeta_score(ytr, oof >= t, beta=2, zero_division=0) for t in thresholds]
    threshold = float(thresholds[np.argmax(f2)])
    probabilities, fitted, rows = {}, {}, []
    for name, pipe in pipes.items():
        print(f"Fitting and evaluating {name} ...", flush=True)
        fitted[name] = clone(pipe).fit(Xtr, ytr)
        probabilities[name] = fitted[name].predict_proba(Xte)[:, 1]
        rows.append(dict(model=name, **metrics(yte, probabilities[name])))
    test = pd.DataFrame(rows)
    intervals = bootstrap_auc(yte, probabilities, gte, winner)
    selected = metrics(yte, probabilities[winner], threshold)
    cv.to_csv(out / "cross_validation.csv", index=False)
    test.to_csv(out / "test_metrics.csv", index=False)
    metadata = dict(selected_model=winner, risk_threshold=threshold, test_metrics_at_selected_threshold=selected,
        bootstrap=intervals, python=platform.python_version(), sklearn=sklearn.__version__,
        seed=SEED, threshold_objective="training out-of-fold non-survival F2",
        selection_objective="training cross-validation mean non-survival average precision")
    (out / "results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    modeldir = Path(args.model_dir)
    modeldir.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(pipeline=fitted[winner], threshold=threshold, metadata=metadata), modeldir / "best_model.joblib")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, p in probabilities.items():
        RocCurveDisplay.from_predictions(yte, p, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(yte, p, name=name, ax=axes[1])
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].axhline(yte.mean(), color="gray", linestyle="--")
    axes[0].set_title("Held-out ROC: non-survival positive")
    axes[1].set_title("Held-out precision-recall")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    summary = cv.merge(test, on="model")
    table = markdown_table(summary[["model", "cv_ap", "cv_auc", "roc_auc", "average_precision", "risk_recall", "risk_precision", "brier"]])
    report = f"""# Evaluation report

## Objective and outcome definition
Predict the supplied survival outcome. Assume `Survive=1/Yes` means survived and `0/No` means did not survive. Internally, non-survival is positive, so risk recall measures detection of non-survivors. The CSV has no data dictionary, follow-up duration or censoring indicator. These are classification estimates, not survival times or probabilities at a defined clinical horizon. Confirm label meaning with the dataset owner before interpreting clinically.

## Data quality
There are {audit['rows']:,} rows, {audit['unique_patient_ids']:,} IDs, {audit['survival_count']:,} survival labels and {audit['non_survival_count']:,} non-survival labels. {audit['negative_age_count']} negative ages are set missing, not converted to positive ages. Creatinine has {audit['missing_after_cleaning']['Creatinine']} missing values. Whitespace/case and ejection-fraction abbreviations are normalized. Nonpositive numeric measurements and ages above 120 are treated as invalid; no additional physiological cutoffs are guessed without units.

{audit['ids_with_conflicting_outcomes']} IDs have conflicting outcomes. Records are retained because timestamps/provenance are unavailable; repeated IDs and exact cleaned predictor copies are joined into groups. Grouping mitigates overlap, but cannot resolve label ambiguity or establish that all records are independent patients. ID is excluded as a predictor. Favorite color is excluded because no clinical rationale is supplied. See `data_audit.json` for all counts and source checksum.

## Fair comparison
One predefined grouped stratified split reserves {len(yte):,} rows (about 20%) for testing; {len(ytr):,} rows remain for five-fold grouped stratified cross-validation. Every model uses identical folds and features. Median/mode imputation, scaling and one-hot encoding are fit within each training fold using pipelines. No patient group crosses partitions. Random seed is 42. Hyperparameters are fixed in advance in `candidates()`.

Three classifiers are evaluated: logistic regression (simple linear reference), random forest (nonlinear bagged trees), and histogram gradient boosting (sequential nonlinear trees). A prior-probability dummy baseline tests whether they improve over prevalence alone. Select by training CV mean average precision for non-survival. ROC AUC assesses ranking; Brier score assesses probability error (lower is better). Accuracy alone would conceal class imbalance. CV standard deviations in `cross_validation.csv` describe fold variation, not confidence intervals.

## Results
All threshold-dependent metrics below use 0.50; non-survival is positive. CV AP determines selection; test scores provide final evaluation only.

{table}

![ROC and precision-recall curves](comparison.png)

## Selected model and operating threshold
**{winner}** has the highest mean training CV average precision among the three classifiers. This is the preferred model under the declared ranking criterion, rather than evidence of clinical readiness. Logistic regression remains easier to explain; forest and boosting can capture nonlinear interactions but are less transparent. A small score difference should not be treated as decisive clinical superiority.

For the selected model, a threshold of **{threshold:.2f}** maximizes F2 over a predefined grid using training out-of-fold predictions. F2 weights recall more than precision, reflecting the exercise's preemptive-treatment motivation; this is an illustrative preference, not a medically validated cost ratio. The grid and objective were fixed before test evaluation. OOF F2 is a tuning score, not an unbiased final performance estimate.

At this threshold, held-out non-survival recall is **{selected['risk_recall']:.3f}**, precision **{selected['risk_precision']:.3f}**, survival recall **{selected['survival_recall']:.3f}**, and F2 **{selected['risk_f2']:.3f}**. Confusion counts: {selected['tp']} correctly flagged non-survivors, {selected['fn']} missed non-survivors, {selected['fp']} survivors flagged at risk, and {selected['tn']} correctly recognized survivors. Lower thresholds can improve risk recall while increasing false alarms.

## Uncertainty and limitations
`results.json` includes 500 paired patient-group bootstrap samples for held-out AUC intervals and selected-minus-comparator differences. If a difference interval crosses zero, this test set does not clearly distinguish the AUCs. Intervals condition on the fitted models and this split; they exclude training and model-selection uncertainty and do not assess significance of AP differences.

The tree models score unusually close to perfect. This may reflect synthetic construction, repeated underlying source records not identifiable by ID, or outcome-derived features; the CSV alone cannot establish which explanation applies. The top two models are effectively tied in practical ranking performance, and the paired AUC interval includes zero. Random forest is the nominal winner under the predeclared CV criterion, not a proven clinical improvement over boosting.

This is an educational dataset with unexplained repeated IDs, conflicting labels and unknown provenance. No external or prospective validation, calibration study, subgroup fairness assessment, treatment-effect analysis or clinical utility evaluation has been performed. These predictions cannot establish which treatment will improve survival. Confirm feature availability before outcome, units, outcome coding and follow-up horizon, then validate independently before considering patient care. The saved model uses training rows only, preserving the reported held-out evaluation.

## Reproduction
Generated using Python {platform.python_version()} and scikit-learn {sklearn.__version__}. See the root README for exact commands. Source SHA-256: `{audit['source_sha256']}`.

Method references: [scikit-learn leakage guidance](https://scikit-learn.org/1.6/common_pitfalls.html#data-leakage), [grouped cross-validation](https://scikit-learn.org/1.6/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    print(table)
    print(f"Selected: {winner}; risk threshold: {threshold:.2f}")


def predict(args):
    artifact = joblib.load(args.model)  # Load only trusted, locally generated artifacts.
    raw = pd.read_csv(args.data)
    risk = artifact["pipeline"].predict_proba(raw)[:, 1]
    result = pd.DataFrame({"survival_probability": 1 - risk, "non_survival_probability": risk,
                           "risk_flag": (risk >= artifact["threshold"]).astype(int)})
    if "ID" in raw:
        result.insert(0, "ID", raw.ID)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    # Use importable class paths in joblib artifacts, including when invoked as a script.
    from survival import train, predict
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/classification_bonus_practice_data.csv")
    t.add_argument("--output", default="reports")
    t.add_argument("--model-dir", default="models")
    t.add_argument("--jobs", type=int, default=2)
    p = sub.add_parser("predict")
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="models/best_model.joblib")
    p.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    train(args) if args.command == "train" else predict(args)
