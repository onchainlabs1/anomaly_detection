"""
Data preprocessing pipeline for the Anomaliq system.
Handles feature scaling, encoding, and preprocessing for anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path

from src.utils import get_settings, get_model_config, model_logger, save_model, load_model


class AnomalyPreprocessor(BaseEstimator, TransformerMixin):
    """Custom preprocessor for anomaly detection data."""
    
    def __init__(
        self,
        scaler_type: str = "standard",
        handle_outliers: bool = True,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5
    ):
        self.scaler_type = scaler_type
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        
        self.scaler = None
        self.outlier_bounds = {}
        self.feature_names = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the preprocessor on training data."""
        model_logger.info("Fitting anomaly preprocessor")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Initialize scaler
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        # Fit scaler
        self.scaler.fit(X)
        
        # Calculate outlier bounds if needed
        if self.handle_outliers:
            self.outlier_bounds = self._calculate_outlier_bounds(X)
        
        self.is_fitted = True
        model_logger.info("Anomaly preprocessor fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X_transformed = X.copy()
        
        # Handle outliers if configured
        if self.handle_outliers:
            X_transformed = self._handle_outliers(X_transformed)
        
        # Apply scaling
        X_scaled = self.scaler.transform(X_transformed)
        
        # Convert back to DataFrame
        X_transformed = pd.DataFrame(
            X_scaled,
            columns=self.feature_names,
            index=X.index
        )
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit preprocessor and transform data."""
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled data back to original scale."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        
        X_original = self.scaler.inverse_transform(X)
        
        return pd.DataFrame(
            X_original,
            columns=self.feature_names,
            index=X.index
        )
    
    def _calculate_outlier_bounds(self, X: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Calculate outlier bounds for each feature."""
        bounds = {}
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if self.outlier_method == "iqr":
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
            elif self.outlier_method == "zscore":
                mean = X[col].mean()
                std = X[col].std()
                lower_bound = mean - self.outlier_threshold * std
                upper_bound = mean + self.outlier_threshold * std
            else:
                # Use percentiles
                lower_bound = X[col].quantile(0.01)
                upper_bound = X[col].quantile(0.99)
            
            bounds[col] = (lower_bound, upper_bound)
        
        return bounds
    
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the data."""
        X_handled = X.copy()
        
        for col, (lower_bound, upper_bound) in self.outlier_bounds.items():
            if col in X_handled.columns:
                # Clip outliers to bounds
                X_handled[col] = X_handled[col].clip(lower_bound, upper_bound)
        
        return X_handled
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on variance after scaling."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        # Use scaler's scale_ attribute (inverse of standard deviation)
        if hasattr(self.scaler, 'scale_'):
            importance = 1.0 / self.scaler.scale_
            importance = importance / importance.sum()  # Normalize
            
            return dict(zip(self.feature_names, importance))
        
        return {}
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save the fitted preprocessor."""
        save_model(self, file_path)
        model_logger.info(f"Preprocessor saved to: {file_path}")
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'AnomalyPreprocessor':
        """Load a fitted preprocessor."""
        preprocessor = load_model(file_path)
        model_logger.info(f"Preprocessor loaded from: {file_path}")
        return preprocessor


class TimeSeriesPreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for time-series features in transaction data."""
    
    def __init__(self, time_column: str = "Time"):
        self.time_column = time_column
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the time series preprocessor."""
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform time features."""
        X_transformed = X.copy()
        
        if self.time_column in X_transformed.columns:
            # Extract time-based features
            time_seconds = X_transformed[self.time_column]
            
            # Hour of day (0-23)
            X_transformed['hour_of_day'] = (time_seconds / 3600) % 24
            
            # Day of week (cyclical encoding)
            day_of_week = (time_seconds / (24 * 3600)) % 7
            X_transformed['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            X_transformed['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            
            # Hour cyclical encoding
            X_transformed['hour_sin'] = np.sin(2 * np.pi * X_transformed['hour_of_day'] / 24)
            X_transformed['hour_cos'] = np.cos(2 * np.pi * X_transformed['hour_of_day'] / 24)
            
            # Business hours indicator
            X_transformed['is_business_hours'] = (
                (X_transformed['hour_of_day'] >= 9) & 
                (X_transformed['hour_of_day'] < 17)
            ).astype(int)
            
            # Weekend indicator
            X_transformed['is_weekend'] = (day_of_week >= 5).astype(int)
            
            # Remove the original time column
            X_transformed = X_transformed.drop(columns=[self.time_column])
        
        return X_transformed


class FeaturePipeline:
    """Complete feature engineering pipeline for anomaly detection."""
    
    def __init__(
        self,
        include_time_features: bool = True,
        scaler_type: str = "standard",
        handle_outliers: bool = True
    ):
        self.include_time_features = include_time_features
        self.scaler_type = scaler_type
        self.handle_outliers = handle_outliers
        
        self.pipeline = None
        self.is_fitted = False
        self.feature_names_in = None
        self.feature_names_out = None
        
        self.settings = get_settings()
    
    def build_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Build the preprocessing pipeline."""
        steps = []
        
        # Add time series preprocessing if requested
        if self.include_time_features and 'Time' in X.columns:
            steps.append(('time_features', TimeSeriesPreprocessor()))
        
        # Add main preprocessing
        steps.append((
            'preprocessor',
            AnomalyPreprocessor(
                scaler_type=self.scaler_type,
                handle_outliers=self.handle_outliers
            )
        ))
        
        pipeline = Pipeline(steps)
        return pipeline
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature pipeline."""
        model_logger.info("Fitting feature pipeline")
        
        self.feature_names_in = list(X.columns)
        self.pipeline = self.build_pipeline(X)
        self.pipeline.fit(X, y)
        
        # Get output feature names after transformation
        X_transformed = self.pipeline.transform(X)
        self.feature_names_out = list(X_transformed.columns)
        
        self.is_fitted = True
        model_logger.info(f"Feature pipeline fitted: {len(self.feature_names_in)} -> {len(self.feature_names_out)} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        return self.pipeline.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit pipeline and transform features."""
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame, partial: bool = True) -> pd.DataFrame:
        """Inverse transform features (partial support)."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before inverse_transform")
        
        # Only the main preprocessor supports inverse transform
        if len(self.pipeline.steps) > 1 and partial:
            preprocessor = self.pipeline.named_steps['preprocessor']
            return preprocessor.inverse_transform(X)
        else:
            model_logger.warning("Full inverse transform not supported with time features")
            return X
    
    def get_feature_names(self) -> List[str]:
        """Get output feature names."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        return self.feature_names_out
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save the fitted pipeline."""
        save_model(self.pipeline, file_path)
        
        # Save metadata
        metadata = {
            'feature_names_in': self.feature_names_in,
            'feature_names_out': self.feature_names_out,
            'include_time_features': self.include_time_features,
            'scaler_type': self.scaler_type,
            'handle_outliers': self.handle_outliers
        }
        
        metadata_path = Path(file_path).with_suffix('.metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        model_logger.info(f"Feature pipeline saved to: {file_path}")
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'FeaturePipeline':
        """Load a fitted pipeline."""
        # Load pipeline
        pipeline_obj = load_model(file_path)
        
        # Load metadata
        metadata_path = Path(file_path).with_suffix('.metadata.json')
        metadata = {}
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Create instance
        instance = cls(
            include_time_features=metadata.get('include_time_features', True),
            scaler_type=metadata.get('scaler_type', 'standard'),
            handle_outliers=metadata.get('handle_outliers', True)
        )
        
        instance.pipeline = pipeline_obj
        instance.feature_names_in = metadata.get('feature_names_in', [])
        instance.feature_names_out = metadata.get('feature_names_out', [])
        instance.is_fitted = True
        
        model_logger.info(f"Feature pipeline loaded from: {file_path}")
        return instance 