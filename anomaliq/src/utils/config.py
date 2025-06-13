"""
Configuration management for the Anomaliq system.
Handles environment variables and application settings using Pydantic.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_debug: bool = Field(default=True, env="API_DEBUG")
    
    # JWT Authentication
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./anomaliq.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # MLflow Configuration
    mlflow_tracking_uri: str = Field(default="./mlruns", env="MLFLOW_TRACKING_URI")
    mlflow_experiment_name: str = Field(default="anomaly_detection", env="MLFLOW_EXPERIMENT_NAME")
    mlflow_model_name: str = Field(default="isolation_forest_anomaly_detector", env="MLFLOW_MODEL_NAME")
    
    # Streamlit Configuration
    streamlit_host: str = Field(default="0.0.0.0", env="STREAMLIT_HOST")
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    
    # Model Configuration
    model_threshold: float = Field(default=0.1, env="MODEL_THRESHOLD")
    feature_columns: str = Field(
        default="V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount",
        env="FEATURE_COLUMNS"
    )
    target_column: str = Field(default="Class", env="TARGET_COLUMN")
    
    # Monitoring Configuration
    drift_threshold: float = Field(default=0.1, env="DRIFT_THRESHOLD")
    alert_email: str = Field(default="admin@anomaliq.com", env="ALERT_EMAIL")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Data Configuration
    data_path: str = Field(default="./data/", env="DATA_PATH")
    reference_data_path: str = Field(default="./data/reference_data.csv", env="REFERENCE_DATA_PATH")
    live_data_path: str = Field(default="./data/live_data.csv", env="LIVE_DATA_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def feature_columns_list(self) -> List[str]:
        """Convert feature columns string to list."""
        return [col.strip() for col in self.feature_columns.split(",")]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    def get_database_url(self) -> str:
        """Get database URL with proper formatting."""
        return self.database_url
    
    def get_mlflow_tracking_uri(self) -> str:
        """Get MLflow tracking URI."""
        return self.mlflow_tracking_uri


class ModelConfig:
    """Configuration for ML model parameters."""
    
    # Isolation Forest parameters
    CONTAMINATION = 0.1
    N_ESTIMATORS = 100
    MAX_SAMPLES = "auto"
    MAX_FEATURES = 1.0
    BOOTSTRAP = False
    RANDOM_STATE = 42
    
    # Feature scaling
    SCALER_TYPE = "standard"  # "standard", "minmax", "robust"
    
    # SHAP configuration
    SHAP_BACKGROUND_SAMPLES = 100
    SHAP_MAX_DISPLAY = 20
    
    # Training configuration
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    CV_FOLDS = 5


class MonitoringConfig:
    """Configuration for monitoring and alerting."""
    
    # Drift detection
    DRIFT_DETECTION_METHODS = ["ks", "wasserstein", "psi"]
    ALERT_THRESHOLD = 0.05
    
    # Performance monitoring
    PERFORMANCE_WINDOW_DAYS = 7
    MIN_SAMPLES_FOR_DRIFT = 1000
    
    # Alerting
    ALERT_COOLDOWN_HOURS = 1
    MAX_ALERTS_PER_DAY = 10


# Global settings instance
settings = Settings()
model_config = ModelConfig()
monitoring_config = MonitoringConfig()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


def get_model_config() -> ModelConfig:
    """Get model configuration instance."""
    return model_config


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration instance."""
    return monitoring_config 