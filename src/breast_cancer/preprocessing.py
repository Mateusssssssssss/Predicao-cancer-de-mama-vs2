import pandas as pd
from sklearn.model_selection import train_test_split

from breast_cancer.config import RANDOM_STATE, TARGET_COLUMN, TEST_SIZE
from breast_cancer.data import clean_data


def prepare_data(frame: pd.DataFrame, test_size: float = TEST_SIZE,
                 random_state: int = RANDOM_STATE):
    """Codifica B=0/M=1 e retorna split estratificado em X_train, X_test, y_train, y_test."""
    frame = clean_data(frame)
    labels = frame.pop(TARGET_COLUMN).map({"B": 0, "M": 1})
    if labels.isna().any():
        raise ValueError("A coluna diagnosis deve conter somente 'B' e 'M'.")
    features = frame
    return train_test_split(
        features, labels.astype("int8"), test_size=test_size,
        random_state=random_state, stratify=labels,
    )
