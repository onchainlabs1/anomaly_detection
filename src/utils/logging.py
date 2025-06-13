"""
Logging configuration for the Anomaliq system.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger
from datetime import datetime


def setup_logger(name: str) -> logger:
    """Set up a logger with the given name."""
    # Remove default logger
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        level="INFO",
        enqueue=True,
        serialize=True
    )
    
    # Add file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
        enqueue=True,
        serialize=True
    )
    
    # Bind name to logger
    return logger.bind(name=f"anomaliq.{name}")


# Create loggers for different components
data_logger = setup_logger("data")
model_logger = setup_logger("model")
api_logger = setup_logger("api")


def log_model_training_start(
    experiment_name: str,
    model_type: str
) -> None:
    """Log the start of model training."""
    model_logger.info(
        "Model training started",
        experiment_name=experiment_name,
        model_type=model_type
    )


def log_model_training_complete(metrics: Dict[str, Any]) -> None:
    """Log the completion of model training."""
    model_logger.info(
        "Model training completed",
        metrics=metrics
    )


def log_api_request(
    endpoint: str,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    response_code: Optional[int] = None
) -> None:
    """Log an API request."""
    api_logger.info(
        "API request",
        endpoint=endpoint,
        method=method,
        params=params,
        response_code=response_code
    )


def log_prediction(
    model_name: str,
    input_shape: tuple,
    prediction: Any,
    confidence: Optional[float] = None
) -> None:
    """Log a model prediction."""
    model_logger.info(
        "Model prediction",
        model_name=model_name,
        input_shape=input_shape,
        prediction=prediction,
        confidence=confidence
    )


def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """Log an error with context."""
    logger.error(
        "Error occurred",
        error=str(error),
        error_type=type(error).__name__,
        context=context
    ) 