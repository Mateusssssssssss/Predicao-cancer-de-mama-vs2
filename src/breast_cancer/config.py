from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "raw" / "cancer_mama.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
IMAGES_DIR = ROOT_DIR / "images"
CANDIDATE_MODEL_DIR = ROOT_DIR / "models" / "candidates"
BEST_MODEL_DIR = ROOT_DIR / "models" / "best_model"
BEST_MODEL_PATH = BEST_MODEL_DIR / "xgboost_best.joblib"
REPORT_DIR = ROOT_DIR / "reports"
OOF_REPORT_DIR = REPORT_DIR / "oof"
VALIDATION_REPORT_DIR = REPORT_DIR / "validation"
FIGURE_DIR = REPORT_DIR / "figures"
TARGET_COLUMN = "diagnosis"
TRAIN_SIZE = 0.6
VALIDATION_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42
THRESHOLD_BETA = 2.0

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
