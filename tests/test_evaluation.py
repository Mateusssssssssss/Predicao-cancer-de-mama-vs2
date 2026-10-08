# NumPy monta os dados numéricos de entrada para as métricas.
import numpy as np

# Função que calcula métricas e relatório de avaliação.
from breast_cancer.evaluation import evaluate_model
# Utilitários de validação cruzada e seleção/análise de threshold.
from breast_cancer.validation import (cross_validated_probabilities,
                                      select_threshold, threshold_analysis)
# Classificador simples usado para exercitar a geração de probabilidades OOF.
from sklearn.dummy import DummyClassifier


def test_evaluation_includes_classification_metrics_and_auprc():
    """Confere métricas, relatório por classe e matriz de confusão."""
    # Rótulos verdadeiros e probabilidades perfeitamente separáveis neste exemplo.
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])

    # Avalia as probabilidades convertendo-as em classes pelo threshold 0.5.
    metrics = evaluate_model(y_true, probabilities, threshold=0.5)

    # Com separação perfeita, accuracy e área precision-recall devem valer 1.
    assert metrics["accuracy"] == 1.0
    assert metrics["auprc"] == 1.0
    # O relatório deve incluir as métricas importantes para cada classe.
    assert "precision" in metrics["classification_report"]
    assert "recall" in metrics["classification_report"]
    assert "f1-score" in metrics["classification_report"]
    # Linhas são classes verdadeiras e colunas são classes previstas.
    assert metrics["confusion_matrix"].tolist() == [[2, 0], [0, 2]]


def test_oof_probabilities_cover_each_row_once():
    """Confere formato e faixa das probabilidades geradas fora dos folds."""
    # Cria 12 amostras sintéticas, duas variáveis e classes alternadas.
    X = np.arange(24).reshape(12, 2)
    y = np.array([0, 1] * 6)
    # DummyClassifier fornece probabilidades de referência sem treino complexo.
    model = DummyClassifier(strategy="prior")

    # Em 3 folds, cada linha recebe uma probabilidade OOF de um modelo que não a viu.
    probabilities = cross_validated_probabilities(model, X, y, cv=3)

    # Esperamos exatamente uma saída por observação, entre 0 e 1.
    assert probabilities.shape == (12,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_threshold_analysis_marks_selected_threshold():
    """Confere que a análise identifica o threshold que maximiza F2."""
    # Exemplo com dois positivos e dois negativos em pontos distintos.
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    # Seleciona o threshold pelo F2 e gera a tabela de métricas para vários limiares.
    threshold = select_threshold(y_true, probabilities, beta=2.0)
    analysis = threshold_analysis(y_true, probabilities, beta=2.0)

    # 0.6 é o primeiro limiar que classifica corretamente ambos os positivos.
    assert threshold == 0.6
    # A tabela deve marcar somente uma linha como limiar escolhido e conter F2.
    assert analysis["selected_threshold"].sum() == 1
    assert "f2_score" in analysis


def test_f2_can_choose_recall_over_a_higher_precision_operating_point():
    """Demonstra que β=2 pode preferir recall total a precision mais alta."""
    # O limiar 0.5 detecta ambos os positivos; thresholds mais altos perdem um deles.
    y_true = np.array([0, 1, 0, 1])
    probabilities = np.array([0.4, 0.5, 0.6, 0.9])

    # Escolhe e consulta as métricas do ponto operacional com melhor F2.
    threshold = select_threshold(y_true, probabilities, beta=2.0)
    analysis = threshold_analysis(y_true, probabilities, beta=2.0)
    selected = analysis.loc[analysis["selected_threshold"]].iloc[0]

    # A escolha recupera todos os positivos, ainda que isso reduza precision.
    assert threshold == 0.5
    assert selected["recall"] == 1.0
    assert selected["precision"] == 2 / 3
