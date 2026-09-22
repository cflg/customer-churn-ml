"""Constantes compartidas del proyecto: columnas, paths y semilla."""
from pathlib import Path

RANDOM_STATE = 42

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "customer_churn_historical.csv"
TRAIN_DATA_PATH = ROOT_DIR / "data" / "processed" / "train.csv"
TEST_DATA_PATH = ROOT_DIR / "data" / "processed" / "test.csv"

TARGET_COL = "Churn"
ID_COL = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
