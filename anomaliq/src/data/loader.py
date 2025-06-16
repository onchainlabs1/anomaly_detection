"""
Data loading utilities for the Anomaliq system.
Handles loading, validation, and preprocessing of datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

from src.utils import (
    get_settings, 
    data_logger, 
    validate_dataframe, 
    clean_column_names,
    handle_missing_values,
    create_feature_summary
)


class DataLoader:
    """Handles loading and validation of datasets."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = data_logger
    
    def load_data(self) -> pd.DataFrame:
        """Load the credit card dataset."""
        data_path = Path(self.settings.data_dir) / "creditcard.csv"
        self.logger.info(f"Loading credit card dataset from: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            self.logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Clean column names
            df = clean_column_names(df)
            
            # Handle any missing values
            if df.isnull().any().any():
                self.logger.warning("Found missing values, applying mean imputation")
                df = handle_missing_values(df, strategy="mean")
            
            return df
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {data_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def load_csv(
        self, 
        file_path: Union[str, Path],
        clean_names: bool = True,
        handle_missing: bool = True,
        missing_strategy: str = "mean"
    ) -> pd.DataFrame:
        """Load CSV file with optional preprocessing."""
        self.logger.info(f"Loading CSV file: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded dataset with shape: {df.shape}")
            
            # Clean column names if requested
            if clean_names:
                df = clean_column_names(df)
                self.logger.info("Column names cleaned")
            
            # Handle missing values if requested
            if handle_missing and df.isnull().any().any():
                self.logger.warning(f"Found missing values, applying strategy: {missing_strategy}")
                df = handle_missing_values(df, strategy=missing_strategy)
                self.logger.info("Missing values handled")
            
            return df
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    def load_training_data(
        self, 
        file_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Load training data with validation."""
        if file_path is None:
            file_path = Path(self.settings.data_path) / "training_data.csv"
        
        df = self.load_csv(file_path)
        
        # Validate required columns for training
        required_columns = self.settings.feature_columns_list
        if self.settings.target_column:
            required_columns.append(self.settings.target_column)
        
        try:
            validate_dataframe(df, required_columns, allow_missing=True)
        except ValueError as e:
            self.logger.warning(f"Validation warning: {str(e)}")
            # If target column is missing, it might be unlabeled data
            if self.settings.target_column not in df.columns:
                self.logger.info("Target column not found - assuming unlabeled data")
        
        self.logger.info("Training data loaded and validated successfully")
        return df
    
    def load_inference_data(
        self, 
        file_path: Union[str, Path]
    ) -> pd.DataFrame:
        """Load data for inference/prediction."""
        df = self.load_csv(file_path)
        
        # Validate feature columns
        available_features = [col for col in self.settings.feature_columns_list if col in df.columns]
        missing_features = set(self.settings.feature_columns_list) - set(available_features)
        
        if missing_features:
            self.logger.warning(f"Missing features for inference: {missing_features}")
        
        # Select only available feature columns
        df = df[available_features]
        
        self.logger.info(f"Inference data loaded with {len(available_features)} features")
        return df
    
    def load_reference_data(self) -> pd.DataFrame:
        """Load reference data for drift monitoring."""
        file_path = self.settings.reference_data_path
        
        if not Path(file_path).exists():
            self.logger.warning(f"Reference data file not found: {file_path}")
            return pd.DataFrame()
        
        df = self.load_csv(file_path)
        self.logger.info("Reference data loaded for drift monitoring")
        return df
    
    def load_live_data(self) -> pd.DataFrame:
        """Load live/production data for drift monitoring."""
        file_path = self.settings.live_data_path
        
        if not Path(file_path).exists():
            self.logger.warning(f"Live data file not found: {file_path}")
            return pd.DataFrame()
        
        df = self.load_csv(file_path)
        self.logger.info("Live data loaded for drift monitoring")
        return df
    
    def validate_feature_schema(
        self, 
        df: pd.DataFrame, 
        reference_columns: List[str]
    ) -> Dict[str, any]:
        """Validate feature schema against reference."""
        current_columns = set(df.columns)
        reference_columns = set(reference_columns)
        
        validation_result = {
            "is_valid": True,
            "missing_columns": list(reference_columns - current_columns),
            "extra_columns": list(current_columns - reference_columns),
            "dtype_mismatches": []
        }
        
        if validation_result["missing_columns"]:
            validation_result["is_valid"] = False
            self.logger.error(f"Missing columns: {validation_result['missing_columns']}")
        
        if validation_result["extra_columns"]:
            self.logger.warning(f"Extra columns found: {validation_result['extra_columns']}")
        
        return validation_result
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """Get comprehensive data summary."""
        summary = create_feature_summary(df)
        
        # Add anomaly-specific insights
        summary["anomaly_insights"] = {}
        
        # Check for potential data quality issues
        for col in df.select_dtypes(include=[np.number]).columns:
            col_data = df[col]
            
            # Check for extreme values
            q1, q99 = col_data.quantile([0.01, 0.99])
            extreme_values = ((col_data < q1) | (col_data > q99)).sum()
            
            summary["anomaly_insights"][col] = {
                "extreme_values_count": extreme_values,
                "extreme_values_percentage": (extreme_values / len(df)) * 100,
                "zero_values_count": (col_data == 0).sum(),
                "zero_values_percentage": ((col_data == 0).sum() / len(df)) * 100
            }
        
        return summary
    
    def prepare_features_target(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features and target from dataset."""
        # Get available feature columns
        available_features = [col for col in self.settings.feature_columns_list if col in df.columns]
        
        if not available_features:
            raise ValueError("No feature columns found in dataset")
        
        X = df[available_features].copy()
        
        # Get target if available
        y = None
        if self.settings.target_column and self.settings.target_column in df.columns:
            y = df[self.settings.target_column].copy()
        
        self.logger.info(f"Prepared features: {len(available_features)} columns")
        if y is not None:
            self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y


class DataValidator:
    """Validates data quality and integrity."""
    
    def __init__(self):
        self.logger = data_logger
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """Comprehensive data quality check."""
        quality_report = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_data": {},
            "duplicates": 0,
            "data_types": {},
            "outliers": {},
            "quality_score": 0.0
        }
        
        # Check missing data
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            quality_report["missing_data"][col] = {
                "count": missing_count,
                "percentage": (missing_count / len(df)) * 100
            }
        
        # Check duplicates
        quality_report["duplicates"] = df.duplicated().sum()
        
        # Check data types
        quality_report["data_types"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Check outliers for numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            quality_report["outliers"][col] = {
                "count": outliers,
                "percentage": (outliers / len(df)) * 100
            }
        
        # Calculate quality score (0-100)
        total_missing = sum(info["count"] for info in quality_report["missing_data"].values())
        missing_penalty = (total_missing / (len(df) * len(df.columns))) * 100
        duplicate_penalty = (quality_report["duplicates"] / len(df)) * 100
        
        quality_report["quality_score"] = max(0, 100 - missing_penalty - duplicate_penalty)
        
        self.logger.info(f"Data quality score: {quality_report['quality_score']:.2f}")
        
        return quality_report
    
    def validate_for_training(self, df: pd.DataFrame) -> bool:
        """Validate dataset is suitable for training."""
        quality_report = self.check_data_quality(df)
        
        # Define validation rules
        min_quality_score = 75.0
        max_missing_percentage = 20.0
        min_rows = 1000
        
        issues = []
        
        if quality_report["quality_score"] < min_quality_score:
            issues.append(f"Quality score too low: {quality_report['quality_score']:.2f}")
        
        if len(df) < min_rows:
            issues.append(f"Insufficient data: {len(df)} rows (minimum: {min_rows})")
        
        # Check excessive missing data
        for col, info in quality_report["missing_data"].items():
            if info["percentage"] > max_missing_percentage:
                issues.append(f"Column '{col}' has {info['percentage']:.1f}% missing data")
        
        if issues:
            self.logger.error(f"Validation failed: {'; '.join(issues)}")
            return False
        
        self.logger.info("Dataset validation passed")
        return True 