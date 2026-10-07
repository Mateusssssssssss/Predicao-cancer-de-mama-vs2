from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "cancer_mama.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
TARGET_COLUMN = "diagnosis"
TEST_SIZE = 0.4
RANDOM_STATE = 42
THRESHOLD_MIN_PRECISION = 0.90

XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 30,
    "subsample": 0.5,
    "colsample_bytree": 0.5,
    "gamma": 1,
    "reg_lambda": 0,
    "reg_alpha": 1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
