# NumPy cria os arrays de rótulos usados no modelo de exemplo.
import numpy as np
# pandas permite testar o suporte a DataFrames e à ordem das colunas.
import pandas as pd
# Regressão logística é usada como modelo leve e determinístico no teste de predição.
from sklearn.linear_model import LogisticRegression

# Funções que criam os modelos candidatos do projeto.
from breast_cancer.models import make_candidates, make_random_forest, make_xgboost
# Funções de inferência que serão verificadas.
from breast_cancer.prediction import predict, predict_proba


def test_factories_build_expected_candidate_models():
    """Verifica se as fábricas retornam os candidatos configurados no projeto."""
    # Constrói o conjunto completo de candidatos.
    candidates = make_candidates()

    # O projeto deve disponibilizar Random Forest e XGBoost.
    assert set(candidates) == {"random_forest", "xgboost"}
    # As fábricas individuais devem retornar o mesmo tipo de modelo do conjunto.
    assert isinstance(make_xgboost(), type(candidates["xgboost"]))
    assert isinstance(make_random_forest(), type(candidates["random_forest"]))


def test_prediction_uses_feature_order_and_threshold():
    """Confere alinhamento das colunas, probabilidades e aplicação do threshold."""
    # As colunas aparecem intencionalmente numa ordem diferente da usada no ajuste.
    X = pd.DataFrame({"right": [0, 1, 2, 3], "left": [3, 2, 1, 0]})
    # Rótulos binários simples para o modelo aprender uma separação clara.
    y = np.array([0, 0, 1, 1])
    # Ajusta a regressão logística na ordem left/right.
    model = LogisticRegression().fit(X[["left", "right"]], y)

    # A função deve reordenar as variáveis usando feature_names.
    probabilities = predict_proba(model, X, feature_names=["left", "right"])
    # Converte as probabilidades em classes usando o limiar 0.5.
    labels = predict(model, X, threshold=0.5, feature_names=["left", "right"])

    # Uma probabilidade por linha e classes previstas conforme os rótulos esperados.
    assert probabilities.shape == (4,)
    assert labels.tolist() == [0, 0, 1, 1]
