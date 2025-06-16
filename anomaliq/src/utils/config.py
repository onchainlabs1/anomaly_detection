"""
Configuration management for the Anomaliq system.
Handles environment variables and application settings using Pydantic.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = Field(default=True, env="API_DEBUG")
    
    # JWT Authentication
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./anomaliq.db", env="DATABASE_URL")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # MLflow Configuration
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = Field(default="anomaly_detection", env="MLFLOW_EXPERIMENT_NAME")
    mlflow_model_name: str = "isolation_forest_anomaly_detector"
    mlflow_model_stage: str = "Production"
    
    # Streamlit Configuration
    streamlit_host: str = Field(default="0.0.0.0", env="STREAMLIT_HOST")
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    
    # Model Configuration
    model_threshold: float = Field(default=0.1, env="MODEL_THRESHOLD")
    feature_columns: str = Field(
        default="v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,time,amount",
        env="FEATURE_COLUMNS"
    )
    target_column: str = Field(default="class", env="TARGET_COLUMN")
    
    # Monitoring Configuration
    drift_threshold: float = Field(default=0.05, env="DRIFT_THRESHOLD")
    alert_email: str = Field(default="admin@anomaliq.com", env="ALERT_EMAIL")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    # Data Configuration
    data_dir: Path = Path(__file__).parent.parent.parent / "src" / "data"
    model_dir: Path = Path(__file__).parent.parent.parent / "models"
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


class ModelConfig(BaseSettings):
    """Configuration for ML model parameters."""
    
    # Isolation Forest parameters
    n_estimators: int = 100
    max_samples: float = 0.8
    contamination: float = 0.1
    random_state: int = 42
    
    # Feature scaling
    scaler_type: str = "standard"  # "standard", "minmax", "robust"
    
    # SHAP configuration
    shap_background_samples: int = 100
    shap_max_display: int = 20
    
    # Training configuration
    test_size: float = 0.2
    validation_size: float = 0.2
    cv_folds: int = 5


class MonitoringConfig(BaseSettings):
    """Configuration for monitoring and alerting."""
    
    # Drift detection
    drift_detection_methods: List[str] = ["ks", "wasserstein", "psi"]
    alert_threshold: float = 0.05
    
    # Performance monitoring
    window_size: int = 1000
    check_interval: int = 100
    
    # Alerting
    alert_cooldown_hours: int = 1
    max_alerts_per_day: int = 10


# Global settings instance
settings = Settings()
model_config = ModelConfig()
monitoring_config = MonitoringConfig()


@lru_cache()
def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


@lru_cache()
def get_model_config() -> ModelConfig:
    """Get model configuration instance."""
    return model_config


@lru_cache()
def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration instance."""
    return monitoring_config 