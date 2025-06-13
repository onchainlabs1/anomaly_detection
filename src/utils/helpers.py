"""
Helper functions for the Anomaliq system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import warnings


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by removing quotes and special characters."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.strip('"').str.strip("'")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean"
) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    df = df.copy()
    
    if strategy == "mean":
        df = df.fillna(df.mean())
    elif strategy == "median":
        df = df.fillna(df.median())
    elif strategy == "mode":
        df = df.fillna(df.mode().iloc[0])
    elif strategy == "drop":
        df = df.dropna()
    else:
        raise ValueError(f"Invalid missing value strategy: {strategy}")
    
    return df


def create_feature_summary(df: pd.DataFrame) -> Dict[str, Dict[str, Union[float, int]]]:
    """Create a summary of feature statistics."""
    summary = {}
    
    for col in df.columns:
        col_stats = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "missing": int(df[col].isnull().sum())
        }
        summary[col] = col_stats
    
    return summary


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: Optional[int] = None,
    max_missing_ratio: Optional[float] = None
) -> bool:
    """Validate a DataFrame against basic requirements."""
    # Check if DataFrame is empty
    if df.empty:
        warnings.warn("DataFrame is empty")
        return False
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            warnings.warn(f"Missing required columns: {missing_cols}")
            return False
    
    # Check minimum number of rows
    if min_rows and len(df) < min_rows:
        warnings.warn(f"DataFrame has fewer than {min_rows} rows")
        return False
    
    # Check missing value ratio
    if max_missing_ratio:
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > max_missing_ratio:
            warnings.warn(f"Missing value ratio {missing_ratio:.2f} exceeds threshold {max_missing_ratio}")
            return False
    
    return True


def ensure_dir(path: Union[str, Path]) -> None:
    """Ensure a directory exists, creating it if necessary."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    """Get current timestamp in a consistent format."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S") 