from pathlib import Path

import pandas as pd

from breast_cancer.config import DATA_PATH, TARGET_COLUMN


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Carrega o CSV e valida a presença da coluna alvo."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em '{path}'. Coloque cancer_mama.csv em data/raw/."
        )
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"O CSV precisa conter a coluna '{TARGET_COLUMN}'.")
    return frame


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove identificadores e colunas vazias conhecidas do dataset original."""
    return frame.drop(columns=["id", "Unnamed: 32"], errors="ignore").copy()
