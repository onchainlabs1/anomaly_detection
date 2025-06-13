"""
Logging configuration for the Anomaliq system.
Provides structured logging with different formats and levels.
"""

import sys
import json
from typing import Any, Dict
from loguru import logger
from .config import get_settings


def format_record(record: Dict[str, Any]) -> str:
    """Format log record as JSON."""
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    
    # Add extra fields if present
    if "extra" in record:
        log_entry.update(record["extra"])
    
    return json.dumps(log_entry)


def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()
    
    # Remove default logger
    logger.remove()
    
    # Configure format based on settings
    if settings.log_format.lower() == "json":
        # JSON format for production
        logger.add(
            sys.stdout,
            format=format_record,
            level=settings.log_level,
            serialize=False,
        )
    else:
        # Human-readable format for development
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            level=settings.log_level,
        )
    
    # Add file logging for production
    if settings.is_production:
        logger.add(
            "logs/anomaliq.log",
            rotation="1 day",
            retention="30 days",
            level=settings.log_level,
            format=format_record,
            serialize=False,
        )


def get_logger(name: str) -> Any:
    """Get a logger instance with the specified name."""
    return logger.bind(name=name)


# Application-specific loggers
api_logger = get_logger("anomaliq.api")
model_logger = get_logger("anomaliq.model")
data_logger = get_logger("anomaliq.data")
dashboard_logger = get_logger("anomaliq.dashboard")
monitoring_logger = get_logger("anomaliq.monitoring")


# Context managers for structured logging
class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, **context):
        self.context = context
        self.token = None
    
    def __enter__(self):
        self.token = logger.contextualize(**self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            self.token.__exit__(exc_type, exc_val, exc_tb)


def log_function_call(func_name: str, **kwargs):
    """Log function call with parameters."""
    return LogContext(
        function_call=func_name,
        parameters=kwargs
    )


def log_model_metrics(model_name: str, metrics: Dict[str, float]):
    """Log model performance metrics."""
    model_logger.info(
        "Model metrics recorded",
        model_name=model_name,
        metrics=metrics
    )


def log_api_request(method: str, endpoint: str, status_code: int, response_time: float):
    """Log API request details."""
    api_logger.info(
        "API request processed",
        method=method,
        endpoint=endpoint,
        status_code=status_code,
        response_time_ms=response_time * 1000
    )


def log_data_drift(feature: str, drift_score: float, threshold: float):
    """Log data drift detection results."""
    monitoring_logger.warning(
        "Data drift detected",
        feature=feature,
        drift_score=drift_score,
        threshold=threshold,
        alert=drift_score > threshold
    )


def log_anomaly_prediction(record_id: str, anomaly_score: float, is_anomaly: bool):
    """Log anomaly prediction results."""
    model_logger.info(
        "Anomaly prediction made",
        record_id=record_id,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly
    )


def log_model_training_start(experiment_name: str, model_type: str):
    """Log model training start."""
    model_logger.info(
        "Model training started",
        experiment_name=experiment_name,
        model_type=model_type
    )


def log_model_training_complete(experiment_name: str, model_type: str, metrics: Dict[str, float]):
    """Log model training completion."""
    model_logger.info(
        "Model training completed",
        experiment_name=experiment_name,
        model_type=model_type,
        final_metrics=metrics
    )


def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log error with context."""
    logger.error(
        f"Error occurred: {str(error)}",
        error_type=type(error).__name__,
        context=context or {},
        exc_info=True
    )


# Initialize logging on import
setup_logging() 