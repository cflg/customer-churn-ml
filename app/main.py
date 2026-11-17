"""Aplicación FastAPI para inferencia en tiempo real de Customer Churn."""
from contextlib import asynccontextmanager
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.inference import predict_single
from app.logging_config import get_logger, log_event
from app.model_loader import ModelArtifact, load_model_artifact
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

logger = get_logger("churn_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una única vez al iniciar el proceso y registra eventos."""
    try:
        artifact = load_model_artifact()
        app.state.model_artifact = artifact
        log_event(
            logger,
            logging.INFO,
            "Servicio iniciado y modelo cargado en memoria",
            event="service_startup",
            model_name=artifact.model_name,
            model_version=artifact.model_version,
            run_id=artifact.run_id,
            source=artifact.source,
        )
    except Exception as exc:
        app.state.model_artifact = None
        log_event(
            logger,
            logging.ERROR,
            "Error al cargar el modelo durante el inicio del servicio",
            event="startup_error",
            error=str(exc),
        )
    yield
    app.state.model_artifact = None
    log_event(
        logger,
        logging.INFO,
        "Servicio detenido y recursos liberados",
        event="service_shutdown",
    )


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Servicio REST de inferencia para estimar probabilidad y nivel de riesgo "
        "de abandono de clientes en telecomunicaciones (ISTEA - MLOps)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Asigna request_id, mide latencia de procesamiento y registra cada solicitud."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        log_event(
            logger,
            logging.ERROR,
            "Fallo no controlado procesando solicitud",
            event="unhandled_exception",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            latency_ms=latency_ms,
            error=str(exc),
        )
        raise

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    request.state.latency_ms = latency_ms
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(latency_ms)

    log_event(
        logger,
        logging.INFO,
        "Solicitud HTTP procesada",
        event="http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=latency_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Registra errores de contrato (422) con su respectivo request_id."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    errors = exc.errors()
    log_event(
        logger,
        logging.WARNING,
        "Solicitud rechazada por validación de contrato Pydantic",
        event="validation_error",
        request_id=request_id,
        path=request.url.path,
        error_count=len(errors),
    )
    return JSONResponse(
        status_code=422,
        content={"detail": errors, "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


def _get_loaded_artifact(request: Request) -> ModelArtifact:
    """Verifica que el artefacto del modelo esté cargado; devuelve 503 si no lo está."""
    artifact: ModelArtifact | None = getattr(
        request.app.state, "model_artifact", None
    )
    if artifact is None or artifact.pipeline is None:
        request_id = getattr(request.state, "request_id", "unknown")
        log_event(
            logger,
            logging.ERROR,
            "Intento de uso con modelo no disponible en memoria",
            event="model_unavailable",
            request_id=request_id,
            path=request.url.path,
        )
        raise HTTPException(
            status_code=503,
            detail="Servicio no disponible: el modelo no está cargado en memoria.",
        )
    return artifact


@app.get("/health", response_model=HealthResponse, tags=["Operación"])
def health(request: Request) -> HealthResponse:
    """Verifica disponibilidad operativa del servicio y versión del modelo cargado."""
    artifact = _get_loaded_artifact(request)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=artifact.model_name,
        model_version=artifact.model_version,
        run_id=artifact.run_id,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inferencia"])
def predict(
    payload: PredictionRequest, request: Request
) -> PredictionResponse:
    """Valida los datos de un cliente y ejecuta el pipeline completo de inferencia."""
    artifact = _get_loaded_artifact(request)
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    infer_start = time.perf_counter()
    try:
        result = predict_single(
            pipeline=artifact.pipeline,
            request=payload,
            model_version=artifact.model_version,
            request_id=request_id,
        )
    except Exception as exc:
        log_event(
            logger,
            logging.ERROR,
            "Error durante la ejecución del pipeline de inferencia",
            event="inference_error",
            request_id=request_id,
            model_version=artifact.model_version,
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail="Error interno al ejecutar la predicción.",
        ) from exc

    inference_ms = round((time.perf_counter() - infer_start) * 1000, 2)
    log_event(
        logger,
        logging.INFO,
        "Predicción de churn ejecutada exitosamente",
        event="inference_prediction",
        request_id=request_id,
        prediction=result["prediction"],
        probability=result["probability"],
        risk_level=result["risk_level"],
        model_version=result["model_version"],
        run_id=artifact.run_id,
        inference_ms=inference_ms,
    )
    return PredictionResponse(**result)
