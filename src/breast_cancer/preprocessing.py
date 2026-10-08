from math import isclose

import pandas as pd
from sklearn.model_selection import train_test_split

from breast_cancer.config import (RANDOM_STATE, TARGET_COLUMN, TEST_SIZE,
                                  TRAIN_SIZE, VALIDATION_SIZE)
from breast_cancer.data import clean_data


def separate_features_target(frame: pd.DataFrame):
    """Limpa o frame e retorna X e y com B=0 e M=1."""
    frame = clean_data(frame)
    labels = frame.pop(TARGET_COLUMN).map({"B": 0, "M": 1})
    if labels.isna().any():
        raise ValueError("A coluna diagnosis deve conter somente 'B' e 'M'.")
    return frame, labels.astype("int8")


def prepare_data(frame: pd.DataFrame, train_size: float = TRAIN_SIZE,
                 validation_size: float = VALIDATION_SIZE,
                 test_size: float = TEST_SIZE,
                 random_state: int = RANDOM_STATE):
    """Retorna split estratificado em treino, validação e teste, nessa ordem."""
    if not isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train_size, validation_size e test_size devem somar 1.0.")
    features, labels = separate_features_target(frame)
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        features, labels, train_size=train_size,
        random_state=random_state, stratify=labels,
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_remaining, y_remaining, test_size=test_size / (validation_size + test_size),
        random_state=random_state, stratify=y_remaining,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test
