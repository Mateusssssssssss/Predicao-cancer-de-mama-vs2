import pandas as pd
import pytest

from breast_cancer.preprocessing import prepare_data, separate_features_target


def sample_frame():
    return pd.DataFrame({
        "id": range(8),
        "diagnosis": ["B", "M"] * 4,
        "feature_a": range(8),
        "feature_b": [2, 3] * 4,
        "Unnamed: 32": [None] * 8,
    })


def test_separate_features_target_cleans_and_encodes_labels():
    features, labels = separate_features_target(sample_frame())

    assert list(features.columns) == ["feature_a", "feature_b"]
    assert labels.tolist() == [0, 1] * 4


def test_prepare_data_returns_stratified_train_test_split():
    X_train, X_test, y_train, y_test = prepare_data(sample_frame(), test_size=0.25)

    assert len(X_train) == len(y_train) == 6
    assert len(X_test) == len(y_test) == 2
    assert y_train.mean() == y_test.mean() == 0.5


def test_invalid_diagnosis_is_rejected():
    frame = sample_frame()
    frame.loc[0, "diagnosis"] = "unknown"

    with pytest.raises(ValueError, match="somente 'B' e 'M'"):
        separate_features_target(frame)
