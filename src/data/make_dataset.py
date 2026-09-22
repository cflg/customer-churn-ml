"""
Carga el dataset crudo, hace la limpieza mínima necesaria y genera
la partición train/test que va a usar el resto del pipeline.

Uso:
    python -m src.data.make_dataset
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    RANDOM_STATE,
    TARGET_COL,
)


def load_raw(path=RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # TotalCharges viene como string y tiene vacíos (clientes con tenure 0).
    # Se fuerza a numérico; los que no matchean quedan NaN y los resuelve
    # el imputer del pipeline, no acá.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    return df


def split(df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COL],
    )
    return train_df, test_df


def main():
    df = load_raw()
    train_df, test_df = split(df)

    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(f"train: {train_df.shape} -> {TRAIN_DATA_PATH}")
    print(f"test:  {test_df.shape} -> {TEST_DATA_PATH}")
    print("target rate train:", train_df[TARGET_COL].value_counts(normalize=True).to_dict())
    print("target rate test: ", test_df[TARGET_COL].value_counts(normalize=True).to_dict())


if __name__ == "__main__":
    main()
