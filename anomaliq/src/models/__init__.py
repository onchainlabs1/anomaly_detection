"""
Models package for the Anomaliq system.
Handles model training, inference, and prediction.
"""

from .train_model import AnomalyModelTrainer
from .predictor import AnomalyPredictor

__all__ = [
    "AnomalyModelTrainer",
    "AnomalyPredictor",
] 