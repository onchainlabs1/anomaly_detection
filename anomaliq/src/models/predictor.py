"""
Model inference and prediction for the Anomaliq system.
Handles model loading, prediction, and SHAP explanations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import mlflow
import mlflow.sklearn
import shap
import joblib

from src.features import FeaturePipeline
from src.utils import (
    get_settings, get_model_config, model_logger,
    log_anomaly_prediction, load_model
)


class AnomalyPredictor:
    """Handles anomaly prediction with explainability."""
    
    def __init__(self, model_name: Optional[str] = None, model_path: Optional[str] = None):
        self.settings = get_settings()
        self.model_config = get_model_config()
        self.logger = model_logger
        
        self.model = None
        self.feature_pipeline = None
        self.shap_explainer = None
        self.background_data = None
        
        # Load model
        if model_name:
            self.load_model_by_name(model_name)
        elif model_path:
            self.load_model_from_path(model_path)
    
    def load_model_by_name(self, model_name: str, version: str = "latest") -> None:
        """Load model from MLflow model registry."""
        self.logger.info(f"Loading model '{model_name}' version '{version}' from MLflow")
        
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
            
            # Load model from registry
            model_uri = f"models:/{model_name}/{version}"
            self.model = mlflow.sklearn.load_model(model_uri)
            
            # Try to load associated feature pipeline
            pipeline_path = f"models/{model_name}_pipeline.joblib"
            if Path(pipeline_path).exists():
                self.feature_pipeline = FeaturePipeline.load(pipeline_path)
            
            self.logger.info("Model loaded successfully from MLflow")
            
        except Exception as e:
            self.logger.error(f"Failed to load model from MLflow: {str(e)}")
            raise
    
    def load_model_from_path(self, model_path: str) -> None:
        """Load model from local file path."""
        self.logger.info(f"Loading model from path: {model_path}")
        
        try:
            self.model = load_model(model_path)
            
            # Try to load associated feature pipeline
            pipeline_path = Path(model_path).parent / f"{Path(model_path).stem}_pipeline.joblib"
            if pipeline_path.exists():
                self.feature_pipeline = FeaturePipeline.load(pipeline_path)
            
            self.logger.info("Model loaded successfully from file")
            
        except Exception as e:
            self.logger.error(f"Failed to load model from file: {str(e)}")
            raise
    
    def setup_shap_explainer(self, background_data: Optional[pd.DataFrame] = None) -> None:
        """Setup SHAP explainer for model interpretability."""
        if self.model is None:
            raise ValueError("Model must be loaded before setting up SHAP explainer")
        
        self.logger.info("Setting up SHAP explainer")
        
        try:
            if background_data is not None:
                # Use provided background data
                if self.feature_pipeline:
                    self.background_data = self.feature_pipeline.transform(background_data)
                else:
                    self.background_data = background_data
            else:
                # Create synthetic background data if none provided
                self.logger.warning("No background data provided, creating synthetic background")
                self.background_data = self._create_synthetic_background()
            
            # Sample background data if too large
            if len(self.background_data) > self.model_config.SHAP_BACKGROUND_SAMPLES:
                self.background_data = self.background_data.sample(
                    n=self.model_config.SHAP_BACKGROUND_SAMPLES,
                    random_state=self.model_config.RANDOM_STATE
                )
            
            # Create SHAP explainer
            self.shap_explainer = shap.Explainer(
                self.model.decision_function,
                self.background_data
            )
            
            self.logger.info("SHAP explainer setup successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup SHAP explainer: {str(e)}")
            self.shap_explainer = None
    
    def _create_synthetic_background(self) -> pd.DataFrame:
        """Create synthetic background data for SHAP."""
        # This is a simple implementation - in practice, you'd use real training data
        n_features = len(self.settings.feature_columns_list)
        n_samples = self.model_config.SHAP_BACKGROUND_SAMPLES
        
        # Generate synthetic data
        data = np.random.normal(0, 1, (n_samples, n_features))
        
        return pd.DataFrame(data, columns=self.settings.feature_columns_list[:n_features])
    
    def predict_single(
        self, 
        record: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        explain: bool = True
    ) -> Dict[str, Any]:
        """Predict anomaly for a single record."""
        if self.model is None:
            raise ValueError("Model must be loaded before making predictions")
        
        # Convert input to DataFrame
        if isinstance(record, dict):
            df = pd.DataFrame([record])
        elif isinstance(record, pd.Series):
            df = pd.DataFrame([record])
        elif isinstance(record, pd.DataFrame):
            df = record.copy()
        else:
            raise ValueError("Record must be dict, Series, or DataFrame")
        
        # Apply feature pipeline if available
        if self.feature_pipeline:
            df_processed = self.feature_pipeline.transform(df)
        else:
            df_processed = df
        
        # Get prediction
        anomaly_score = self.model.decision_function(df_processed)[0]
        is_anomaly = anomaly_score < -self.settings.model_threshold
        
        # Prepare result
        result = {
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'threshold': self.settings.model_threshold,
            'confidence': float(abs(anomaly_score))
        }
        
        return result
    
    def predict_batch(
        self, 
        data: pd.DataFrame,
        explain: bool = False,
        return_details: bool = False
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Predict anomalies for a batch of records."""
        if self.model is None:
            raise ValueError("Model must be loaded before making predictions")
        
        self.logger.info(f"Making batch predictions for {len(data)} records")
        
        # Apply feature pipeline if available
        if self.feature_pipeline:
            data_processed = self.feature_pipeline.transform(data)
        else:
            data_processed = data
        
        # Get predictions
        anomaly_scores = self.model.decision_function(data_processed)
        is_anomaly = anomaly_scores < -self.settings.model_threshold
        
        if not return_details:
            return anomaly_scores
        
        # Prepare detailed results
        result = {
            'anomaly_scores': anomaly_scores.tolist(),
            'is_anomaly': is_anomaly.tolist(),
            'threshold': self.settings.model_threshold,
            'n_anomalies': int(np.sum(is_anomaly)),
            'anomaly_rate': float(np.mean(is_anomaly))
        }
        
        # Add SHAP explanations if requested
        if explain and self.shap_explainer is not None:
            try:
                self.logger.info("Computing SHAP explanations for batch")
                shap_values = self.shap_explainer(data_processed)
                
                # Extract values
                if hasattr(shap_values, 'values'):
                    shap_vals = shap_values.values
                else:
                    shap_vals = shap_values
                
                # Add to result
                result['shap_values'] = shap_vals.tolist()
                result['feature_names'] = data_processed.columns.tolist()
                
            except Exception as e:
                self.logger.warning(f"Batch SHAP explanation failed: {str(e)}")
                result['shap_values'] = None
        
        self.logger.info(f"Batch prediction completed: {result['n_anomalies']} anomalies detected")
        return result
    
    def get_feature_importance(self, data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Get overall feature importance from the model or SHAP."""
        if self.model is None:
            raise ValueError("Model must be loaded first")
        
        if self.shap_explainer is not None and data is not None:
            # Use SHAP for feature importance
            try:
                if self.feature_pipeline:
                    data_processed = self.feature_pipeline.transform(data)
                else:
                    data_processed = data
                
                shap_values = self.shap_explainer(data_processed)
                
                # Calculate mean absolute SHAP values
                if hasattr(shap_values, 'values'):
                    mean_shap = np.mean(np.abs(shap_values.values), axis=0)
                else:
                    mean_shap = np.mean(np.abs(shap_values), axis=0)
                
                feature_names = data_processed.columns.tolist()
                return dict(zip(feature_names, mean_shap))
                
            except Exception as e:
                self.logger.warning(f"SHAP feature importance failed: {str(e)}")
        
        # Fallback to model-specific feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            if self.feature_pipeline:
                feature_names = self.feature_pipeline.get_feature_names()
            else:
                feature_names = self.settings.feature_columns_list
            
            return dict(zip(feature_names, self.model.feature_importances_))
        
        return {}
    
    def update_threshold(self, new_threshold: float) -> None:
        """Update the anomaly detection threshold."""
        self.logger.info(f"Updating threshold from {self.settings.model_threshold} to {new_threshold}")
        self.settings.model_threshold = new_threshold
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {}
        
        info = {
            'model_type': type(self.model).__name__,
            'has_feature_pipeline': self.feature_pipeline is not None,
            'has_shap_explainer': self.shap_explainer is not None,
            'threshold': self.settings.model_threshold
        }
        
        # Add model-specific parameters if available
        if hasattr(self.model, 'get_params'):
            info['model_params'] = self.model.get_params()
        
        # Add feature pipeline info if available
        if self.feature_pipeline:
            info['feature_pipeline_info'] = {
                'input_features': len(self.feature_pipeline.feature_names_in or []),
                'output_features': len(self.feature_pipeline.feature_names_out or []),
                'scaler_type': getattr(self.feature_pipeline, 'scaler_type', 'unknown')
            }
        
        return info 