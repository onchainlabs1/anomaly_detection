"""
Data validation utilities for the Anomaliq system.
Validates datasets for training and inference.
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


class DataValidator:
    """Validates datasets for training and inference."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = data_logger
    
    def check_data_quality(self, df: pd.DataFrame) -> float:
        """Check data quality and return a quality score."""
        # Calculate quality metrics
        total_rows = len(df)
        non_null_rows = df.notna().all(axis=1).sum()
        quality_score = (non_null_rows / total_rows) * 100
        
        self.logger.info(f"Data quality score: {quality_score:.2f}")
        return quality_score
    
    def validate_for_training(self, df: pd.DataFrame) -> bool:
        """Validate dataset for training."""
        try:
            # Check data quality
            quality_score = self.check_data_quality(df)
            if quality_score < self.settings.data_quality_threshold:
                self.logger.error(f"Data quality score {quality_score:.2f} below threshold {self.settings.data_quality_threshold}")
                return False
            
            # Check required columns
            required_columns = set(self.settings.feature_columns_list)
            if self.settings.target_column:
                required_columns.add(self.settings.target_column)
            
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check data types
            numeric_columns = df[self.settings.feature_columns_list].select_dtypes(include=[np.number]).columns
            if len(numeric_columns) != len(self.settings.feature_columns_list):
                non_numeric = set(self.settings.feature_columns_list) - set(numeric_columns)
                self.logger.error(f"Non-numeric feature columns: {non_numeric}")
                return False
            
            # Check target column if available
            if self.settings.target_column and self.settings.target_column in df.columns:
                target_values = df[self.settings.target_column].unique()
                if not all(isinstance(x, (int, np.integer)) for x in target_values):
                    self.logger.error("Target column must contain integer values")
                    return False
                if not set(target_values).issubset({0, 1}):
                    self.logger.error("Target column must contain only binary values (0 or 1)")
                    return False
            
            self.logger.info("Dataset validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Dataset validation failed: {str(e)}")
            return False 