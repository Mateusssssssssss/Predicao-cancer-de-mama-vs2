from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from breast_cancer.config import RANDOM_STATE, XGBOOST_PARAMS


def make_xgboost() -> XGBClassifier:
    return XGBClassifier(**XGBOOST_PARAMS)


def make_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=1000, random_state=RANDOM_STATE, n_jobs=-1)


def make_candidates() -> dict:
    """Instancia os modelos candidatos do projeto."""
    return {
        "random_forest": make_random_forest(),
        "xgboost": make_xgboost(),
    }
