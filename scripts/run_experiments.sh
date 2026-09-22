#!/usr/bin/env bash
# Corre el set de experimentos base para la Entrega 1.
# Requiere que antes se haya corrido: python -m src.data.make_dataset
set -e

python -m src.training.train --model baseline --run-name baseline_most_frequent

python -m src.training.train --model logreg --C 0.1 --run-name logreg_C0.1
python -m src.training.train --model logreg --C 1.0 --run-name logreg_C1
python -m src.training.train --model logreg --C 1.0 --balanced --run-name logreg_C1_balanced

python -m src.training.train --model rf --n-estimators 200 --max-depth 6 --run-name rf_200_depth6
python -m src.training.train --model rf --n-estimators 400 --max-depth 12 --balanced --run-name rf_400_depth12_balanced

echo "Listo. Revisar los runs en MLflow y elegir el candidato con: python -m src.training.train --model <elegido> ... --register"
