# customer-churn-ml

Proyecto integrador de Laboratorio de Minería de Datos (ISTEA).

Predicción de churn de clientes de telecomunicaciones. Parte de un flujo reproducible de entrenamiento (DVC + MLflow + scikit-learn Pipeline) y lo expone como servicio REST con FastAPI y Pydantic, testeado con pytest, empaquetado con Docker / Docker Compose y monitoreado con logging estructurado y Evidently.

## Estructura

```
customer-churn-ml/
├── app/
│   ├── main.py              # FastAPI, endpoints /health y /predict, middleware de logs
│   ├── schemas.py           # contratos de entrada y salida con Pydantic
│   ├── model_loader.py      # carga del artefacto y metadata del modelo
│   ├── inference.py         # armado de DataFrame y ejecución del pipeline
│   └── logging_config.py    # formato JSON para logs operativos y de inferencia
├── data/                    # raw, processed y production (no se suben CSV a git)
├── examples/                # payloads de prueba válido e inválido para /predict
├── models/
│   ├── churn_pipeline.joblib
│   └── model_metadata.json
├── monitoring/
│   ├── drift_report.py      # comparación de drift entre histórico y producción con Evidently
│   ├── DRIFT_ANALYSIS.md    # análisis de variables con drift e impacto en el modelo
│   └── reports/
├── notebooks/               # EDA exploratorio
├── scripts/run_experiments.sh
├── src/
│   ├── config.py
│   ├── data/make_dataset.py
│   ├── features/preprocessing.py
│   ├── training/train.py
│   ├── evaluation/metrics.py
│   └── inference/predict.py
├── tests/                   # suite pytest (datos, preprocesamiento, modelo, API y drift)
├── Dockerfile
├── compose.yaml
├── pytest.ini
└── requirements.txt
```

## Setup local

```bash
git clone <url-del-repo>
cd customer-churn-ml
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Datos (DVC + DagsHub)

Los CSV no se suben a git. Se versionan con DVC contra un remote en DagsHub:

```bash
dvc pull
```

Para regenerar el split 80/20 estratificado por `Churn` (`random_state=42`):

```bash
python -m src.data.make_dataset
```

## Experimentos y modelo seleccionado (MLflow)

```bash
bash scripts/run_experiments.sh
```

Métricas sobre el test set (20%, aislado del entrenamiento):

| run                       | modelo    | accuracy | precision | recall | f1    | ROC-AUC |
|---------------------------|-----------|----------|-----------|--------|-------|---------|
| baseline_most_frequent    | Dummy     | 0.736    | 0.000     | 0.000  | 0.000 | 0.500   |
| logreg_C0.1               | LogReg    | 0.791    | 0.656     | 0.441  | 0.527 | 0.812   |
| logreg_C1                 | LogReg    | 0.794    | 0.663     | 0.449  | 0.535 | 0.812   |
| **logreg_C1_balanced**    | **LogReg**| 0.720    | 0.480     | **0.720** | 0.576 | **0.812** |
| rf_200_depth6             | RF        | 0.783    | 0.690     | 0.323  | 0.440 | 0.800   |
| rf_400_depth12_balanced   | RF        | 0.785    | 0.601     | 0.551  | 0.575 | 0.802   |

### Por qué se eligió `logreg_C1_balanced`

El dataset tiene ~26% de churn, por lo que accuracy por sí sola resulta engañosa. En este problema el error más costoso es el **falso negativo**: clasificar como estable a un cliente que estaba por darse de baja impide accionar una campaña de retención a tiempo. Por eso priorizamos **recall** sobre la clase `Yes` manteniendo buen **ROC-AUC**.

`logreg_C1_balanced` logra el mejor recall (0.720) y el mayor ROC-AUC (0.812) con el modelo más simple. Quedó registrado en MLflow Model Registry como `customer-churn-classifier` (versión 1, `run_id=d8ce42b0f1dc4891bfa8e4267b840f0b`) y exportado en `models/churn_pipeline.joblib` junto con `models/model_metadata.json`.

## Testing (pytest)

```bash
pytest -v
```

La suite en `tests/` cubre:
- Esquema de datos y presencia de columnas requeridas (`test_data_schema.py`).
- Pipeline de preprocesamiento, imputación de faltantes en `TotalCharges` y categorías no vistas (`test_preprocessing.py`).
- Carga del artefacto del modelo y su metadata (`test_model_loader.py`).
- Inferencia, rangos de probabilidad y umbrales de `risk_level` (`test_predictor.py`).
- Endpoints `GET /health`, `POST /predict`, documentación OpenAPI `/docs` y rechazo `422` ante payloads inválidos (`test_api.py`).
- Preparación de datos y resumen para monitoreo (`test_monitoring.py`).

## Servicio REST (FastAPI + Pydantic)

Para levantar el servicio localmente sin Docker:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Documentación interactiva Swagger UI: `http://localhost:8000/docs`

### `GET /health`

Verifica disponibilidad del servicio y versión del modelo cargado en memoria:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "customer-churn-classifier",
  "model_version": "1",
  "run_id": "d8ce42b0f1dc4891bfa8e4267b840f0b"
}
```

### `POST /predict`

Recibe las 19 variables del cliente, valida tipos, rangos y categorías permitidas con Pydantic y devuelve predicción, probabilidad, nivel de riesgo (`LOW` < 0.30, `MEDIUM` entre 0.30 y 0.60, `HIGH` >= 0.60), versión del modelo y `request_id`:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: req-demo-01" \
  -d @examples/predict_request_valid.json
```

Respuesta:

```json
{
  "prediction": "Yes",
  "probability": 0.9154,
  "risk_level": "HIGH",
  "model_version": "1",
  "request_id": "req-demo-01"
}
```

Si la entrada es inválida (por ejemplo `examples/predict_request_invalid.json`), responde `422 Unprocessable Entity` con el detalle de los campos rechazados y el `request_id`.

## Docker y Docker Compose (Runbook)

Variables de entorno configurables (con valores por defecto en `compose.yaml`):
- `PORT` (default `8000`)
- `MODEL_PATH` (default `models/churn_pipeline.joblib`)
- `MODEL_METADATA_PATH` (default `models/model_metadata.json`)
- `MODEL_VERSION` (default `1`)
- `LOG_LEVEL` (default `INFO`)

Comandos operativos:

```bash
# Construir imagen
docker build -t customer-churn-ml:entrega-2 .

# Levantar el servicio en segundo plano
docker compose up -d

# Verificar estado y health check
docker compose ps

# Ver logs estructurados
docker compose logs -f churn-api

# Detener el servicio
docker compose down

# Actualizar imagen y reiniciar
docker compose pull && docker compose up -d
```

## Observabilidad y análisis de Data Drift

### Logging y operación

La API emite logs en formato JSON por salida estándar incluyendo `timestamp`, `level`, `event`, `request_id`, `model_version` y tiempos de respuesta (`latency_ms` e `inference_ms`):
- **Servicio caído**: se detecta mediante `GET /health` y el `healthcheck` de Docker (devuelve `503` si el modelo no pudo cargarse en memoria o el contenedor pasa a `unhealthy` si el proceso no responde).
- **Errores HTTP**: cada rechazo de validación (`422`) o fallo interno (`500`/`503`) queda registrado con su `request_id` y causa.
- **Aumento de latencia**: cada request informa `latency_ms` en los logs y en el header `X-Process-Time-Ms`.

### Monitoreo de Data Drift con Evidently

Para comparar los datos históricos de referencia contra el lote de producción (`data/production/customer_churn_current.csv`):

```bash
python -m monitoring.drift_report
```

Esto genera el reporte interactivo en `monitoring/reports/data_drift_report.html` y el resumen en `monitoring/reports/data_drift_summary.json`.

El análisis detallado de las 11 variables con drift (`MonthlyCharges`, `tenure`, `TotalCharges`, `Contract`, `InternetService`, etc.), su efecto sobre las predicciones del modelo y la justificación de por qué no se debe disparar un reentrenamiento automático se encuentran en `monitoring/DRIFT_ANALYSIS.md`.
