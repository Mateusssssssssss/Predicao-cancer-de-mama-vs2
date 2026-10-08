"""Carregamento de artefatos e funções de inferência."""

from pathlib import Path

import joblib
import numpy as np


def load_model_artifact(path: str | Path):
    """Carrega um artefato salvo, validando que contém um modelo."""
    artifact = joblib.load(Path(path))
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("O artefato deve ser um dicionário contendo a chave 'model'.")
    return artifact


def predict_proba(model, X, feature_names=None):
    """Retorna probabilidade da classe positiva, alinhando colunas se necessário."""
    if feature_names is not None:
        X = X.loc[:, feature_names]
    return model.predict_proba(X)[:, 1]


def predict(model, X, threshold: float = 0.5, feature_names=None):
    """Converte probabilidades em rótulos binários usando o threshold informado."""
    return (predict_proba(model, X, feature_names) >= threshold).astype(np.int8)
