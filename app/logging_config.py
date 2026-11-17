"""Configuración de logging estructurado en formato JSON para observabilidad."""
from datetime import datetime, timezone
import json
import logging
import os
import sys
from typing import Any


class StructuredJsonFormatter(logging.Formatter):
    """Formateador JSON para trazabilidad de eventos operativos e inferencia."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "structured_data", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "churn_service") -> logging.Logger:
    """Devuelve un logger configurado para emitir JSON estructurado a stdout."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        logger.propagate = False
    return logger


def log_event(
    logger: logging.Logger, level: int, message: str, **fields: Any
) -> None:
    """Registra un evento con campos estructurados adicionales."""
    logger.log(level, message, extra={"structured_data": fields})
