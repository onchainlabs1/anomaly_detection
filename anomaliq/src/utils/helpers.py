"""
Helper functions and utilities for the Anomaliq system.
Contains common utility functions used across different modules.
"""

import os
import json
import pickle
import hashlib
import time
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import time


def ensure_dir(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_file_hash(file_path: Union[str, Path]) -> str:
    """Get MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_data_hash(data: pd.DataFrame) -> str:
    """Get hash of DataFrame content."""
    return hashlib.md5(
        pd.util.hash_pandas_object(data, index=True).values
    ).hexdigest()


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save dictionary to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """Save object to pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """Load object from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def save_model(model: Any, file_path: Union[str, Path]) -> None:
    """Save model using joblib."""
    joblib.dump(model, file_path)


def load_model(file_path: Union[str, Path]) -> Any:
    """Load model using joblib."""
    return joblib.load(file_path)


def validate_dataframe(
    df: pd.DataFrame, 
    required_columns: List[str],
    allow_missing: bool = False
) -> bool:
    """Validate DataFrame has required columns."""
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns and not allow_missing:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names by removing special characters."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
    return df


def handle_missing_values(
    df: pd.DataFrame, 
    strategy: str = "mean",
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Handle missing values in DataFrame."""
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if df[col].isnull().any():
            if strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif strategy == "zero":
                df[col].fillna(0, inplace=True)
            elif strategy == "drop":
                df.dropna(subset=[col], inplace=True)
    
    return df


def detect_outliers_iqr(
    data: pd.Series, 
    multiplier: float = 1.5
) -> Tuple[pd.Series, pd.Series]:
    """Detect outliers using IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers, data[~outliers]


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """Get memory usage information of DataFrame."""
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    
    return {
        "total_memory": format_bytes(total_memory),
        "per_column": {
            col: format_bytes(memory_usage[col]) 
            for col in df.columns
        }
    }


def sample_dataframe(
    df: pd.DataFrame, 
    n_samples: int, 
    random_state: int = 42,
    stratify_column: Optional[str] = None
) -> pd.DataFrame:
    """Sample DataFrame with optional stratification."""
    if len(df) <= n_samples:
        return df
    
    if stratify_column and stratify_column in df.columns:
        # Stratified sampling
        return df.groupby(stratify_column, group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), max(1, int(n_samples * len(x) / len(df)))),
                random_state=random_state
            )
        ).reset_index(drop=True)
    else:
        # Simple random sampling
        return df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)


def split_dataframe(
    df: pd.DataFrame,
    train_size: float = 0.8,
    random_state: int = 42,
    stratify_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train and test sets."""
    from sklearn.model_selection import train_test_split
    
    if stratify_column and stratify_column in df.columns:
        stratify = df[stratify_column]
    else:
        stratify = None
    
    train_df, test_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def calculate_percentiles(data: pd.Series) -> Dict[str, float]:
    """Calculate common percentiles for a data series."""
    return {
        "min": data.min(),
        "p1": data.quantile(0.01),
        "p5": data.quantile(0.05),
        "p25": data.quantile(0.25),
        "p50": data.quantile(0.50),
        "p75": data.quantile(0.75),
        "p95": data.quantile(0.95),
        "p99": data.quantile(0.99),
        "max": data.max(),
        "mean": data.mean(),
        "std": data.std()
    }


def create_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Create comprehensive feature summary."""
    summary = {
        "shape": df.shape,
        "memory_usage": get_memory_usage(df),
        "columns": {}
    }
    
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "null_count": df[col].isnull().sum(),
            "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
            "unique_count": df[col].nunique()
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_info.update(calculate_percentiles(df[col]))
        
        summary["columns"][col] = col_info
    
    return summary


def retry_with_backoff(
    func,
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    exceptions: Tuple = (Exception,)
):
    """Decorator for retrying functions with exponential backoff."""
    def decorator(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = backoff_factor * (2 ** attempt)
                time.sleep(wait_time)
    
    return decorator


def is_business_hours(dt: datetime = None) -> bool:
    """Check if given datetime is during business hours (9 AM - 5 PM, weekdays)."""
    if dt is None:
        dt = datetime.now()
    
    # Check if it's a weekday (Monday = 0, Sunday = 6)
    if dt.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check if it's during business hours (9 AM - 5 PM)
    return 9 <= dt.hour < 17


def generate_unique_id(prefix: str = "") -> str:
    """Generate unique ID with optional prefix."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_part = hashlib.md5(str(datetime.now().microsecond).encode()).hexdigest()[:8]
    return f"{prefix}{timestamp}_{random_part}" if prefix else f"{timestamp}_{random_part}" 