"""Contratos de entrada y salida de la API implementados con Pydantic."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Contrato de entrada con las 19 features predictivas de un cliente."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
        },
    )

    gender: Literal["Female", "Male"] = Field(
        ..., description="Género registrado del cliente"
    )
    SeniorCitizen: Literal[0, 1] = Field(
        ..., description="Indicador binario de cliente adulto mayor (0 o 1)"
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Indica si el cliente declara pareja"
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Indica si el cliente declara dependientes"
    )
    tenure: int = Field(
        ..., ge=0, le=72, description="Meses de permanencia en la compañía (0 a 72)"
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Indica si posee servicio telefónico"
    )
    MultipleLines: Literal["Yes", "No", "No phone service"] = Field(
        ..., description="Estado de líneas telefónicas múltiples"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Tipo de servicio de Internet contratado"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Servicio de seguridad online"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Servicio de backup online"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Servicio de protección de dispositivos"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Servicio de soporte técnico"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Servicio de streaming de TV"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Servicio de streaming de películas"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Modalidad contractual del cliente"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Indica si utiliza facturación electrónica"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Método habitual de pago")
    MonthlyCharges: float = Field(
        ..., gt=0, le=500, description="Cargo mensual actual (mayor a 0)"
    )
    TotalCharges: Optional[float] = Field(
        default=None,
        ge=0,
        description="Cargos acumulados (puede ser null si tenure=0)",
    )


class PredictionResponse(BaseModel):
    """Contrato de salida de POST /predict."""

    prediction: Literal["Yes", "No"] = Field(
        ..., description="Clase predicha por el modelo (Yes = abandona, No = permanece)"
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probabilidad estimada de abandono [0, 1]"
    )
    risk_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Nivel de riesgo operacional derivado de la probabilidad"
    )
    model_version: str = Field(
        ..., description="Versión del modelo registrado utilizada para inferir"
    )
    request_id: str = Field(
        ..., description="Identificador único de trazabilidad de la solicitud"
    )


class HealthResponse(BaseModel):
    """Contrato de salida de GET /health."""

    status: Literal["ok"] = Field(..., description="Estado operativo del servicio")
    model_loaded: bool = Field(..., description="Indica si el pipeline está en memoria")
    model_name: str = Field(..., description="Nombre del modelo en Model Registry")
    model_version: str = Field(..., description="Versión activa del modelo")
    run_id: str = Field(..., description="Run ID de MLflow que originó el modelo")
