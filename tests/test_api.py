"""Tests de integración y contratos HTTP sobre los endpoints FastAPI."""
from fastapi.testclient import TestClient
import pytest

from app.main import app


def test_health_endpoint_ok(client):
    """GET /health debe responder 200 con estado ok y versión del modelo."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_name"] == "customer-churn-classifier"
    assert data["model_version"] == "1"
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers


def test_predict_endpoint_valid_payload(client, valid_customer_payload):
    """POST /predict con payload válido debe responder 200 y cumplir el contrato."""
    custom_req_id = "req-integration-001"
    response = client.post(
        "/predict",
        json=valid_customer_payload,
        headers={"X-Request-ID": custom_req_id},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Yes", "No"]
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["model_version"] == "1"
    assert data["request_id"] == custom_req_id
    assert response.headers["X-Request-ID"] == custom_req_id


def test_predict_endpoint_allows_null_total_charges(client, valid_customer_payload):
    """Clientes nuevos (tenure=0) pueden enviar TotalCharges=None y ser imputados."""
    payload = dict(valid_customer_payload, tenure=0, TotalCharges=None)
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Yes", "No"]
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_endpoint_official_invalid_payload(client, invalid_customer_payload):
    """El payload inválido de ejemplo debe ser rechazado con HTTP 422."""
    response = client.post("/predict", json=invalid_customer_payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert "request_id" in body


@pytest.mark.parametrize(
    "invalid_override",
    [
        {"tenure": -1},
        {"tenure": 120},
        {"MonthlyCharges": -10.5},
        {"MonthlyCharges": "not-a-number"},
        {"Contract": "Weekly"},
        {"InternetService": "Satellite"},
        {"SeniorCitizen": 5},
        {"unexpected_field": "valor"},
    ],
)
def test_predict_endpoint_rejects_invalid_fields(
    client, valid_customer_payload, invalid_override
):
    """Cualquier violación de dominio, rango o campo extra debe devolver HTTP 422."""
    bad_payload = dict(valid_customer_payload, **invalid_override)
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_endpoint_missing_required_field(client, valid_customer_payload):
    """La omisión de un campo requerido debe devolver HTTP 422."""
    incomplete_payload = dict(valid_customer_payload)
    incomplete_payload.pop("Contract")
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_openapi_swagger_accessible(client):
    """La documentación OpenAPI/Swagger debe estar accesible."""
    docs_res = client.get("/docs")
    openapi_res = client.get("/openapi.json")
    assert docs_res.status_code == 200
    assert openapi_res.status_code == 200
    assert "/predict" in openapi_res.json()["paths"]
    assert "/health" in openapi_res.json()["paths"]


def test_service_unavailable_when_model_missing(valid_customer_payload):
    """Si el modelo no está cargado en memoria, /health y /predict deben devolver 503."""
    with TestClient(app) as test_client:
        original_artifact = test_client.app.state.model_artifact
        try:
            test_client.app.state.model_artifact = None
            health_res = test_client.get("/health")
            predict_res = test_client.post("/predict", json=valid_customer_payload)
            assert health_res.status_code == 503
            assert predict_res.status_code == 503
        finally:
            test_client.app.state.model_artifact = original_artifact
