"""Validação cruzada, busca de hiperparâmetros e seleção de threshold."""

import numpy as np
import pandas as pd
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     cross_val_predict, cross_val_score)

try:
    from sklearn.model_selection import TunedThresholdClassifierCV
except ImportError:  # scikit-learn < 1.5
    TunedThresholdClassifierCV = None


def cross_validated_probabilities(estimator, X, y, cv: int = 5, n_jobs: int = 1):
    """Retorna probabilidades OOF para a classe positiva, uma por amostra."""
    probabilities = cross_val_predict(
        estimator, X, y, cv=cv, method="predict_proba", n_jobs=n_jobs
    )
    classes = np.unique(y)
    positive_column = int(np.flatnonzero(classes == 1)[0])
    return probabilities[:, positive_column]


def select_threshold(y_true, probabilities, beta: float = 2.0) -> float:
    """Maximiza F-beta; beta > 1 dá mais peso ao recall que à precision."""
    from sklearn.metrics import precision_recall_curve

    if beta <= 0:
        raise ValueError("beta deve ser maior que zero.")
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    precision = precision[:-1]
    recall = recall[:-1]
    beta_squared = beta**2
    f_beta = np.divide(
        (1 + beta_squared) * precision * recall,
        beta_squared * precision + recall,
        out=np.zeros_like(precision),
        where=(beta_squared * precision + recall) != 0,
    )
    # Empates de F-beta favorecem maior recall e, depois, maior precision.
    best_index = np.lexsort((precision, recall, f_beta))[-1]
    return float(thresholds[best_index])


def threshold_analysis(y_true, probabilities, beta: float = 2.0) -> pd.DataFrame:
    """Precision, recall, F1 e F-beta por threshold, marcando a escolha por F-beta."""
    from sklearn.metrics import precision_recall_curve

    if beta <= 0:
        raise ValueError("beta deve ser maior que zero.")
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) != 0)
    beta_squared = beta**2
    f_beta = np.divide(
        (1 + beta_squared) * precision * recall,
        beta_squared * precision + recall,
        out=np.zeros_like(precision),
        where=(beta_squared * precision + recall) != 0,
    )
    selected_threshold = select_threshold(y_true, probabilities, beta)
    return pd.DataFrame({
        "threshold": thresholds,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        f"f{beta:g}_score": f_beta,
        "selected_threshold": np.isclose(thresholds, selected_threshold),
    }).sort_values("threshold", ignore_index=True)


def cross_validation_scores(estimator, X, y, scoring: str = "accuracy",
                            cv: int = 5, n_jobs: int = 1):
    """Calcula a métrica escolhida em cada fold da validação cruzada."""
    return cross_val_score(estimator, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs)


def grid_search(estimator, param_grid, X, y, scoring: str = "roc_auc",
                cv: int = 5, n_jobs: int = 1):
    """Executa GridSearchCV e retorna o objeto ajustado."""
    return GridSearchCV(estimator, param_grid, scoring=scoring,
                        cv=cv, n_jobs=n_jobs, refit=True).fit(X, y)


def randomized_search(estimator, param_distributions, X, y, n_iter: int = 20,
                      scoring: str = "roc_auc", cv: int = 5, n_jobs: int = 1,
                      random_state: int = 42):
    """Executa RandomizedSearchCV e retorna o objeto ajustado."""
    return RandomizedSearchCV(
        estimator, param_distributions, n_iter=n_iter, scoring=scoring,
        cv=cv, n_jobs=n_jobs, random_state=random_state, refit=True,
    ).fit(X, y)


def tuned_threshold_classifier(estimator, beta: float = 2.0, cv: int = 5):
    """Cria TunedThresholdClassifierCV quando suportado pela versão instalada."""
    if TunedThresholdClassifierCV is None:
        raise ImportError("TunedThresholdClassifierCV requer scikit-learn >= 1.5.")
    if beta <= 0:
        raise ValueError("beta deve ser maior que zero.")
    from sklearn.metrics import fbeta_score, make_scorer

    scorer = make_scorer(fbeta_score, beta=beta, zero_division=0)
    return TunedThresholdClassifierCV(estimator, scoring=scorer, cv=cv)
