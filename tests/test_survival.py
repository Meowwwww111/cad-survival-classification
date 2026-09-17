import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from survival import CleanFeatures, load_data, make_pipeline


def fixture_frame(n=30):
    return pd.DataFrame({
        'ID': [f'p{i}' for i in range(n)], 'Survive': ['Yes', 'No'] * (n // 2),
        'Gender': ['Male', 'Female'] * (n // 2), 'Smoke': [' Yes ', 'No'] * (n // 2),
        'Diabetes': ['Normal'] * n, 'Age': np.arange(n) + 40,
        'Ejection Fraction': ['L', 'N'] * (n // 2), 'Sodium': [137] * n,
        'Creatinine': [1.2] * n, 'Platelets': [200000] * n,
        'Creatine phosphokinase': [150] * n, 'Blood Pressure': [110] * n,
        'Hemoglobin': [13] * n, 'Height': [170] * n, 'Weight': [70] * n})


def test_cleaning_and_no_mutation():
    raw = fixture_frame()
    raw.loc[0, 'Age'] = -60
    clean = CleanFeatures().transform(raw)
    assert np.isnan(clean.loc[0, 'Age'])
    assert raw.loc[0, 'Age'] == -60
    assert clean.loc[0, 'Smoke'] == 'yes'
    assert clean.loc[0, 'Ejection Fraction'] == 'low'
    assert 'ID' not in clean and 'Survive' not in clean


def test_mapping_and_connected_groups(tmp_path):
    raw = fixture_frame()
    raw.loc[1, 'ID'] = raw.loc[0, 'ID']
    for col in raw.columns.difference(['ID', 'Survive']):
        raw.loc[2, col] = raw.loc[1, col]
    path = tmp_path / 'input.csv'
    raw.to_csv(path, index=False)
    X, y, groups, audit = load_data(path)
    assert list(y[:2]) == [0, 1]
    assert groups[0] == groups[1] == groups[2]
    assert audit['ids_with_conflicting_outcomes'] == 1
    for a, b in StratifiedGroupKFold(3).split(X, y, groups):
        assert not set(groups[a]) & set(groups[b])


def test_unknown_outcome_rejected(tmp_path):
    raw = fixture_frame()
    raw.loc[0, 'Survive'] = 'unknown'
    path = tmp_path / 'input.csv'
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match='label'):
        load_data(path)


def test_imputation_fit_on_training_only_and_unseen_category():
    raw = fixture_frame()
    model = make_pipeline(LogisticRegression(max_iter=1000)).fit(raw.iloc[:20], np.arange(20) % 2)
    imputer = model.named_steps['prepare'].named_transformers_['numeric'].named_steps['impute']
    assert imputer.statistics_[0] == 49.5
    new = raw.iloc[20:].copy()
    new.loc[:, 'Gender'] = 'unseen'
    new['Age'] = np.nan
    p = model.predict_proba(new)
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1)
    assert imputer.statistics_[0] == 49.5
