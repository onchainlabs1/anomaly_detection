"""
Configuration settings for the Anomaliq system.
Uses Pydantic for settings management and validation.
"""

from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
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
    mlflow_experiment_name: str = Field(default="credit_card_fraud_detection", env="MLFLOW_EXPERIMENT_NAME")
    mlflow_model_name: str = Field(default="isolation_forest_anomaly_detector", env="MLFLOW_MODEL_NAME")
    
    # Streamlit Configuration
    streamlit_host: str = Field(default="0.0.0.0", env="STREAMLIT_HOST")
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    
    # Model Configuration
    model_threshold: float = Field(default=0.1, env="MODEL_THRESHOLD")
    model_server_url: str = Field(default="http://localhost:8000", env="MODEL_SERVER_URL")
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
    data_quality_threshold: float = Field(default=0.8, env="DATA_QUALITY_THRESHOLD")
    
    @property
    def feature_columns_list(self) -> List[str]:
        """Get feature columns as a list."""
        return [col.strip() for col in self.feature_columns.split(",")]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings."""
    return Settings() 