"""
Anomaliq - Production-ready anomaly detection system.

A comprehensive MLOps solution for detecting anomalous financial and operational 
records using unsupervised learning techniques. Built with FastAPI, Streamlit, 
MLflow, and SHAP for explainable AI.

Main modules:
- api: FastAPI backend with JWT authentication  
- dashboard: Streamlit interactive dashboard
- models: Model training and inference
- features: Feature engineering and preprocessing
- data: Data loading and synthetic data generation
- utils: Configuration, logging, and utilities
- monitoring: Data drift detection and monitoring
"""

__version__ = "1.0.0"
__author__ = "Anomaliq Team"
__email__ = "team@anomaliq.com"

# Main package imports
from . import api
from . import dashboard
from . import models
from . import features
from . import data
from . import utils
from . import monitoring

__all__ = [
    "api",
    "dashboard", 
    "models",
    "features",
    "data",
    "utils",
    "monitoring",
    "__version__",
] 