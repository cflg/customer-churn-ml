"""Tests de esquema y contrato mínimo de datos."""
import pandas as pd
import pytest

from src.config import (
    FEATURE_COLUMNS,
    ID_COL,
    RAW_DATA_PATH,
    TARGET_COL,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)


@pytest.mark.skipif(
    not RAW_DATA_PATH.exists(), reason="Dataset histórico no presente localmente"
)
def test_historical_dataset_schema():
    """Verifica que el dataset histórico contenga todas las features y el target."""
    df = pd.read_csv(RAW_DATA_PATH, nrows=50)
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    assert TARGET_COL in df.columns
    assert ID_COL not in FEATURE_COLUMNS


@pytest.mark.skipif(
    not (TRAIN_DATA_PATH.exists() and TEST_DATA_PATH.exists()),
    reason="Splits train/test no generados localmente",
)
def test_train_test_splits_schema():
    """Verifica consistencia de columnas y valores de target en train/test."""
    train_df = pd.read_csv(TRAIN_DATA_PATH, nrows=50)
    test_df = pd.read_csv(TEST_DATA_PATH, nrows=50)

    assert set(FEATURE_COLUMNS).issubset(train_df.columns)
    assert set(FEATURE_COLUMNS).issubset(test_df.columns)
    assert set(train_df[TARGET_COL].unique()).issubset({"Yes", "No"})
    assert set(test_df[TARGET_COL].unique()).issubset({"Yes", "No"})
