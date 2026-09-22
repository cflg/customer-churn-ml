# customer-churn-ml

Proyecto integrador de Laboratorio de Minería de Datos (ISTEA) — Entrega 1.

Predicción de churn de clientes de telecomunicaciones. El objetivo de esta
primera etapa es tener un proyecto Python reproducible (no un notebook
suelto), con datos versionados en DVC y experimentos trazables en MLflow.

## Estructura

```
customer-churn-ml/
├── data/               # raw/processed (versionados con DVC, no con git)
├── notebooks/          # EDA exploratorio, no forma parte del flujo productivo
├── src/
│   ├── data/           # carga + split reproducible
│   ├── features/       # ColumnTransformer (impute/encode/scale)
│   ├── training/        # entrenamiento + logging a MLflow
│   ├── evaluation/      # métricas
│   └── inference/       # carga de modelo y predicción (se usa en Entrega 2)
├── tests/               # se completa en Entrega 2
├── monitoring/          # se completa en Entrega Final
├── models/
├── scripts/run_experiments.sh
└── requirements.txt
```

## Setup

```bash
git clone <url-del-repo>
cd customer-churn-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Datos (DVC + DagsHub)

Los CSV no se suben a git. Se versionan con DVC contra un remote en DagsHub.

```bash
dvc init
dvc remote add origin https://dagshub.com/<usuario>/customer-churn-ml.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <usuario-dagshub>
dvc remote modify origin --local password <token-dagshub>   # no se commitea

dvc add data/raw/customer_churn_historical.csv
git add data/raw/customer_churn_historical.csv.dvc data/.gitignore
git commit -m "dvc: track dataset historico"
dvc push
```

Para recuperar los datos en otra máquina: `dvc pull`.

## Experimentos (MLflow + DagsHub)

MLflow apunta a DagsHub por variables de entorno (nada de credenciales
en el código):

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/<usuario>/customer-churn-ml.mlflow
export MLFLOW_TRACKING_USERNAME=<usuario-dagshub>
export MLFLOW_TRACKING_PASSWORD=<token-dagshub>
```

Sin esas variables, MLflow loguea localmente en `./mlruns` (así se probó
este repo antes de conectar DagsHub).

### Generar el split

```bash
python -m src.data.make_dataset
```

Split 80/20 estratificado por `Churn`, `random_state=42` (`src/config.py`).
`TotalCharges` se limpia acá (viene como texto con vacíos en clientes con
`tenure=0`); el resto de la imputación queda en el pipeline de sklearn para
que training e inferencia apliquen exactamente las mismas transformaciones.

### Correr los experimentos

```bash
bash scripts/run_experiments.sh
```

Corre baseline (`DummyClassifier`), tres variantes de `LogisticRegression`
y dos de `RandomForestClassifier` (6 runs). Cada uno también se puede
correr suelto:

```bash
python -m src.training.train --model rf --n-estimators 400 --max-depth 12 --balanced
```

## Resultados

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

Accuracy no sirve como criterio único acá: el dataset tiene ~26% de churn,
así que un modelo que casi no detecta la clase positiva (como el RF sin
balancear, recall 0.32) igual saca accuracy alta.

Lo que más nos importa es el **falso negativo**: un cliente que va a
abandonar y el modelo lo marca como estable no dispara ninguna acción de
retención y se termina yendo. Eso pesa más que un falso positivo (contactar
de más a alguien que en realidad no se iba a ir). Por eso priorizamos
**recall** de la clase `Yes` sin resignar ROC-AUC.

`logreg_C1_balanced` es el que mejor recall tiene (0.72) manteniendo el
ROC-AUC más alto del grupo (0.812) y F1 competitivo con el mejor RF. Es
además el modelo más simple de los seis, lo cual pesa a favor dado que la
consigna no premia sobreingeniería. Es el que quedó registrado en el Model
Registry como `customer-churn-classifier`.

El threshold de decisión (0.5 acá) y los cortes de `risk_level` se van a
terminar de ajustar en la Entrega 2 una vez que el negocio defina el costo
relativo de cada tipo de error.

## Trazabilidad

El run que originó el modelo registrado queda identificado en MLflow
(`run_id`, parámetros, métricas y artefacto `model/`) y linkeado a la
versión del Model Registry. El `run_id` correspondiente a
`customer-churn-classifier` v1 puede verse en la UI de MLflow / DagsHub.

## Reproducir todo desde cero

```bash
git clone <url-del-repo> && cd customer-churn-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
dvc pull
python -m src.data.make_dataset
bash scripts/run_experiments.sh
```
