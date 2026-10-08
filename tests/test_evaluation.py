import numpy as np

from breast_cancer.evaluation import evaluate_model
from breast_cancer.validation import (cross_validated_probabilities,
                                      select_threshold, threshold_analysis)
from sklearn.dummy import DummyClassifier


def test_evaluation_includes_classification_metrics_and_auprc():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])

    metrics = evaluate_model(y_true, probabilities, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["auprc"] == 1.0
    assert "precision" in metrics["classification_report"]
    assert "recall" in metrics["classification_report"]
    assert "f1-score" in metrics["classification_report"]
    assert metrics["confusion_matrix"].tolist() == [[2, 0], [0, 2]]


def test_oof_probabilities_cover_each_row_once():
    X = np.arange(24).reshape(12, 2)
    y = np.array([0, 1] * 6)
    model = DummyClassifier(strategy="prior")

    probabilities = cross_validated_probabilities(model, X, y, cv=3)

    assert probabilities.shape == (12,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_threshold_analysis_marks_selected_threshold():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = select_threshold(y_true, probabilities, beta=2.0)
    analysis = threshold_analysis(y_true, probabilities, beta=2.0)

    assert threshold == 0.6
    assert analysis["selected_threshold"].sum() == 1
    assert "f2_score" in analysis


def test_f2_can_choose_recall_over_a_higher_precision_operating_point():
    y_true = np.array([0, 1, 0, 1])
    probabilities = np.array([0.4, 0.5, 0.6, 0.9])

    threshold = select_threshold(y_true, probabilities, beta=2.0)
    analysis = threshold_analysis(y_true, probabilities, beta=2.0)
    selected = analysis.loc[analysis["selected_threshold"]].iloc[0]

    assert threshold == 0.5
    assert selected["recall"] == 1.0
    assert selected["precision"] == 2 / 3
