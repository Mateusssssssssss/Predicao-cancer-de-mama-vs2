import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from breast_cancer.models import make_candidates, make_random_forest, make_xgboost
from breast_cancer.prediction import predict, predict_proba


def test_factories_build_expected_candidate_models():
    candidates = make_candidates()

    assert set(candidates) == {"random_forest", "xgboost"}
    assert isinstance(make_xgboost(), type(candidates["xgboost"]))
    assert isinstance(make_random_forest(), type(candidates["random_forest"]))


def test_prediction_uses_feature_order_and_threshold():
    X = pd.DataFrame({"right": [0, 1, 2, 3], "left": [3, 2, 1, 0]})
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(X[["left", "right"]], y)

    probabilities = predict_proba(model, X, feature_names=["left", "right"])
    labels = predict(model, X, threshold=0.5, feature_names=["left", "right"])

    assert probabilities.shape == (4,)
    assert labels.tolist() == [0, 0, 1, 1]
