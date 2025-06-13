"""
Monitoring package for the Anomaliq system.
Handles data drift detection and system monitoring.
"""

from .drift_detection import DataDriftDetector

__all__ = [
    "DataDriftDetector",
] 