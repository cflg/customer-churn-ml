"""Fixtures compartidas para la suite de testing de Customer Churn."""
import json

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from app.main import app
from app.model_loader import load_model_artifact
from src.config import ROOT_DIR


@pytest.fixture(scope="session")
def valid_customer_payload() -> dict:
    """Payload válido de ejemplo para un cliente individual."""
    example_path = ROOT_DIR / "examples" / "predict_request_valid.json"
    if example_path.exists():
        with open(example_path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 32,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 101.35,
        "TotalCharges": 2865.88,
    }


@pytest.fixture(scope="session")
def invalid_customer_payload() -> dict:
    """Payload inválido oficial de ejemplo para validar rechazos de contrato."""
    example_path = ROOT_DIR / "examples" / "predict_request_invalid.json"
    if example_path.exists():
        with open(example_path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": -12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Weekly",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": "not-a-number",
        "TotalCharges": 2865.88,
    }


@pytest.fixture(scope="session")
def sample_customer_df(valid_customer_payload) -> pd.DataFrame:
    """DataFrame de una fila con datos válidos de cliente."""
    return pd.DataFrame([valid_customer_payload])


@pytest.fixture(scope="session")
def loaded_artifact():
    """Artefacto del modelo cargado una vez por sesión de tests."""
    return load_model_artifact()


@pytest.fixture()
def client():
    """Cliente de pruebas de FastAPI con ejecución de ciclo de vida (lifespan)."""
    with TestClient(app) as test_client:
        yield test_client
