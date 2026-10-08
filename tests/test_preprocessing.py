# pandas cria os DataFrames usados como entrada dos testes.
import pandas as pd
# pytest fornece as ferramentas para executar testes e verificar erros esperados.
import pytest

# Funções de preparação de dados que serão verificadas neste arquivo.
from breast_cancer.preprocessing import prepare_data, separate_features_target


def sample_frame():
    """Monta um pequeno dataset controlado para testar o pré-processamento."""
    return pd.DataFrame({
        # A coluna id deve ser removida como identificador.
        "id": range(8),
        # Alternar B e M garante exemplos das duas classes em todas as divisões.
        "diagnosis": ["B", "M"] * 4,
        # Atributos artificiais usados para verificar quais colunas permanecem.
        "feature_a": range(8),
        "feature_b": [2, 3] * 4,
        # Coluna vazia semelhante à que pode vir no CSV original.
        "Unnamed: 32": [None] * 8,
    })


def test_separate_features_target_cleans_and_encodes_labels():
    """Confere a remoção de colunas indesejadas e a codificação B/M em 0/1."""
    # Separa os atributos X do alvo y e aplica a limpeza/codificação.
    features, labels = separate_features_target(sample_frame())

    # id e coluna vazia não devem ser usadas como variáveis preditoras.
    assert list(features.columns) == ["feature_a", "feature_b"]
    # Benigno deve ser 0 e maligno deve ser 1, preservando a ordem das linhas.
    assert labels.tolist() == [0, 1] * 4


def test_prepare_data_returns_stratified_train_validation_test_split():
    """Confere tamanhos e proporção das classes nos três conjuntos do split."""
    # A função retorna X e y para treino, validação e teste, nessa ordem.
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(sample_frame())

    # Com oito linhas e a configuração 60/20/20, esperamos 4, 2 e 2 amostras.
    assert len(X_train) == len(y_train) == 4
    assert len(X_validation) == len(y_validation) == 2
    assert len(X_test) == len(y_test) == 2
    # O split estratificado preserva 50% de cada classe em cada subconjunto.
    assert y_train.mean() == y_validation.mean() == y_test.mean() == 0.5


def test_invalid_diagnosis_is_rejected():
    """Garante que rótulos fora de B/M sejam rejeitados com erro claro."""
    # Insere um diagnóstico inválido no dataset sintético.
    frame = sample_frame()
    frame.loc[0, "diagnosis"] = "unknown"

    # O pré-processamento deve interromper a execução em vez de codificar esse valor.
    with pytest.raises(ValueError, match="somente 'B' e 'M'"):
        separate_features_target(frame)
