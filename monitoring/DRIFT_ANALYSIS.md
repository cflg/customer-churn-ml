# Análisis de Data Drift en Producción

Comparación entre el dataset histórico de referencia (`data/raw/customer_churn_historical.csv`, 7043 registros) y el lote de producción (`data/production/customer_churn_current.csv`, 2500 registros) usando Evidently (`DataDriftPreset`).

Para regenerar el reporte HTML y el resumen en JSON:

```bash
python -m monitoring.drift_report
```

## 1. Variables con evidencia de drift

Evidently detectó **Dataset Drift (`dataset_drift = true`)** con **11 de las 19 variables predictivas (57.89%)** superando el umbral de cambio de distribución (distancia de Wasserstein normalizada en numéricas y distancia de Jensen-Shannon >= 0.10 en categóricas).

### Variables numéricas

| Variable | Drift Score (Wasserstein) | Media histórico | Media producción | Diferencia relativa | Lectura |
|---|---|---|---|---|---|
| `MonthlyCharges` | 0.4858 (Drift) | 68.17 | 80.30 | +17.80% | Suba marcada en el cargo mensual promedio. |
| `tenure` | 0.4768 (Drift) | 35.17 | 26.16 | -25.63% | Clientes más nuevos, con menor antigüedad promedio. |
| `TotalCharges` | 0.1702 (Drift) | 2312.08 | 2044.25 | -11.58% | Menor cargo acumulado, explicado por la baja en `tenure`. |
| `SeniorCitizen` | Sin drift | 0.1701 | 0.1748 | +2.76% | Distribución estable entre ambos períodos. |

### Variables categóricas con drift

| Variable | Distancia JS | Categoría principal desplazada | Histórico (%) | Producción (%) | Cambio (pp) |
|---|---|---|---|---|---|
| `StreamingTV` | 0.1192 | `Yes` | 38.68% | 52.64% | +13.96 pp |
| `OnlineSecurity` | 0.1184 | `No` | 45.48% | 59.88% | +14.40 pp |
| `InternetService` | 0.1155 | `Fiber optic` | 44.94% | 58.56% | +13.62 pp |
| `StreamingMovies` | 0.1154 | `Yes` | 38.44% | 51.36% | +12.92 pp |
| `TechSupport` | 0.1123 | `No` | 46.98% | 59.88% | +12.90 pp |
| `Contract` | 0.1062 | `Month-to-month` | 54.11% | 66.52% | +12.41 pp |
| `DeviceProtection` | 0.1021 | `No` | 42.87% | 50.92% | +8.05 pp |
| `OnlineBackup` | 0.1009 | `No` | 43.48% | 50.16% | +6.68 pp |

También se observa crecimiento en `PaymentMethod = Electronic check` (de 33.88% a 43.52%, +9.64 pp) y en `PaperlessBilling = Yes` (de 58.71% a 66.24%, +7.53 pp), aunque levemente por debajo del corte de 0.10.

## 2. Magnitud y relevancia del cambio

El cambio no es aislado en una sola columna sino consistente en todo el perfil del cliente:
- Entran clientes con **menor antigüedad** (-9 meses en promedio) y **mayor proporción de contratos mes a mes** (`Month-to-month` pasa de 54.1% a 66.5%, mientras que `Two year` baja de 24.4% a 13.5%).
- Hay mayor proporción de clientes con **fibra óptica** (+13.6 pp) pero **sin servicios de retención técnica** como `OnlineSecurity` ni `TechSupport`.

## 3. Posible impacto sobre el modelo (`logreg_C1_balanced`)

Al correr el pipeline actual sobre ambos lotes se observa un desplazamiento directo en las salidas del modelo:
- **Probabilidad media predicha**: sube de **0.4211** en referencia a **0.6349** en producción (+50.75%).
- **Tasa de churn predicha (`Yes`)**: sube de **39.54%** a **69.80%** (+30.26 pp).
- **Clientes en riesgo `HIGH` (>= 0.60)**: pasa de **30.97%** a **60.92%**.

Existen dos escenarios técnicos a distinguir:
1. Si la suba de `MonthlyCharges` (de 68.17 a 80.30) se debe a un **ajuste tarifario/inflacionario nominal**, el `StandardScaler` entrenado con la media histórica va a empujar artificialmente las probabilidades hacia arriba y generar **más falsos positivos**.
2. Si el cambio responde a una **campaña comercial real** que captó clientes más volátiles (fibra óptica con contrato mensual y sin soporte), entonces el mayor riesgo predicho es consistente con las relaciones aprendidas en entrenamiento.

## 4. Acción propuesta

Detectar Data Drift **no implica reentrenar automáticamente**. El drift muestra que cambió la distribución de entrada $P(X)$, pero sin la etiqueta real (`Churn`) del nuevo período todavía no sabemos si empeoró la capacidad predictiva ni si cambió la relación entre las variables y el abandono.

Pasos propuestos:
1. **Investigar con negocio**: confirmar si el aumento de `MonthlyCharges` es por actualización de precios o por cambio en el mix de planes vendidos.
2. **Medir performance real cuando cierren las etiquetas**: cruzar las predicciones logueadas (usando `request_id`) contra las bajas reales del período para evaluar `Recall`, `Precision` y `ROC-AUC`.
3. **Recalibrar antes de reentrenar**: si el ranking del modelo (`ROC-AUC`) sigue siendo bueno pero el volumen de casos `HIGH` supera la capacidad del equipo de retención, ajustar los puntos de corte de `RISK_THRESHOLDS` o recalibrar probabilidades.
4. **Reentrenar de forma controlada**: recién cuando se cuente con etiquetas reales consolidadas y se verifique degradación en las métricas, agregar el lote validado a DVC, correr los experimentos en MLflow y registrar una versión 2 en el Model Registry.
