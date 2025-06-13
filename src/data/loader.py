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
            # Load CSV file
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
    
    def prepare_features_target(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features and target from dataset."""
        # Normalize column names for comparison
        df_columns = {col.strip().lower(): col for col in df.columns}
        feature_columns = [col.strip().lower() for col in self.settings.feature_columns_list]
        
        # Debug information
        self.logger.debug("Column name comparison:")
        self.logger.debug(f"DataFrame columns (normalized): {list(df_columns.keys())}")
        self.logger.debug(f"Feature columns (normalized): {feature_columns}")
        
        # Find matching feature columns
        available_features = []
        missing_features = []
        
        for feature in feature_columns:
            if feature in df_columns:
                available_features.append(df_columns[feature])
                self.logger.debug(f"Found feature column: {df_columns[feature]}")
            else:
                missing_features.append(feature)
                self.logger.warning(f"Missing feature column: {feature}")
        
        if not available_features:
            error_msg = (
                "No feature columns found in dataset. "
                f"Expected columns: {self.settings.feature_columns_list}, "
                f"Found columns: {list(df.columns)}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        if missing_features:
            self.logger.warning(f"Missing {len(missing_features)} feature columns: {missing_features}")
        
        self.logger.info(f"Using {len(available_features)} feature columns: {available_features}")
        X = df[available_features].copy()
        
        # Get target if available
        y = None
        target_col = self.settings.target_column.strip().lower()
        
        if target_col in df_columns:
            y = df[df_columns[target_col]].copy()
            self.logger.info(f"Found target column: {df_columns[target_col]}")
            self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        else:
            self.logger.warning(
                f"Target column '{self.settings.target_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        
        return X, y 