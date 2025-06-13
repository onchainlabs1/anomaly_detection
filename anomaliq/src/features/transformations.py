"""
Feature transformations for the Anomaliq system.
Advanced feature engineering techniques for anomaly detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from src.utils import model_logger


class AggregateFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create aggregate features for anomaly detection."""
    
    def __init__(self, window_size: int = 10, aggregation_methods: List[str] = None):
        self.window_size = window_size
        self.aggregation_methods = aggregation_methods or ['mean', 'std', 'min', 'max']
        self.is_fitted = False
        self.logger = model_logger
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the transformer."""
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features by adding aggregate features."""
        X_transformed = X.copy()
        
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            for method in self.aggregation_methods:
                if method == 'mean':
                    X_transformed[f'{col}_rolling_mean'] = X[col].rolling(
                        window=self.window_size, min_periods=1
                    ).mean()
                elif method == 'std':
                    X_transformed[f'{col}_rolling_std'] = X[col].rolling(
                        window=self.window_size, min_periods=1
                    ).std().fillna(0)
                elif method == 'min':
                    X_transformed[f'{col}_rolling_min'] = X[col].rolling(
                        window=self.window_size, min_periods=1
                    ).min()
                elif method == 'max':
                    X_transformed[f'{col}_rolling_max'] = X[col].rolling(
                        window=self.window_size, min_periods=1
                    ).max()
        
        return X_transformed


class InteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """Create interaction features between pairs of variables."""
    
    def __init__(self, max_interactions: int = 50):
        self.max_interactions = max_interactions
        self.feature_pairs = []
        self.is_fitted = False
        self.logger = model_logger
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit and select top feature pairs for interactions."""
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Generate all possible pairs
        from itertools import combinations
        all_pairs = list(combinations(numeric_columns, 2))
        
        # Limit the number of interactions
        if len(all_pairs) > self.max_interactions:
            # Select pairs with highest correlation
            correlations = []
            for col1, col2 in all_pairs:
                corr = abs(X[col1].corr(X[col2]))
                correlations.append((corr, col1, col2))
            
            # Sort by correlation and take top pairs
            correlations.sort(reverse=True)
            self.feature_pairs = [(col1, col2) for _, col1, col2 in correlations[:self.max_interactions]]
        else:
            self.feature_pairs = all_pairs
        
        self.is_fitted = True
        self.logger.info(f"Selected {len(self.feature_pairs)} feature pairs for interactions")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by adding interaction features."""
        X_transformed = X.copy()
        
        for col1, col2 in self.feature_pairs:
            if col1 in X.columns and col2 in X.columns:
                # Multiplication interaction
                X_transformed[f'{col1}_x_{col2}'] = X[col1] * X[col2]
                
                # Division interaction (with zero handling)
                denominator = X[col2].replace(0, np.finfo(float).eps)
                X_transformed[f'{col1}_div_{col2}'] = X[col1] / denominator
        
        return X_transformed


class AnomalyScoreTransformer(BaseEstimator, TransformerMixin):
    """Create anomaly-specific features."""
    
    def __init__(self):
        self.feature_stats = {}
        self.is_fitted = False
        self.logger = model_logger
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit by calculating feature statistics."""
        self.feature_stats = {}
        
        for col in X.select_dtypes(include=[np.number]).columns:
            self.feature_stats[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'median': X[col].median(),
                'q25': X[col].quantile(0.25),
                'q75': X[col].quantile(0.75)
            }
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by adding anomaly-specific features."""
        X_transformed = X.copy()
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if col in self.feature_stats:
                stats = self.feature_stats[col]
                
                # Z-score (standardized distance from mean)
                if stats['std'] > 0:
                    X_transformed[f'{col}_zscore'] = (X[col] - stats['mean']) / stats['std']
                else:
                    X_transformed[f'{col}_zscore'] = 0
                
                # Distance from median
                X_transformed[f'{col}_median_dist'] = abs(X[col] - stats['median'])
                
                # IQR outlier score
                iqr = stats['q75'] - stats['q25']
                if iqr > 0:
                    lower_bound = stats['q25'] - 1.5 * iqr
                    upper_bound = stats['q75'] + 1.5 * iqr
                    X_transformed[f'{col}_iqr_outlier'] = (
                        (X[col] < lower_bound) | (X[col] > upper_bound)
                    ).astype(int)
                else:
                    X_transformed[f'{col}_iqr_outlier'] = 0
        
        return X_transformed


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """Reduce dimensionality while preserving anomaly detection capability."""
    
    def __init__(self, method: str = 'pca', n_components: Optional[int] = None, variance_threshold: float = 0.95):
        self.method = method
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.reducer = None
        self.is_fitted = False
        self.logger = model_logger
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the dimensionality reducer."""
        if self.method == 'pca':
            # Determine number of components if not specified
            if self.n_components is None:
                # Find components that explain variance_threshold of variance
                temp_pca = PCA()
                temp_pca.fit(X)
                cumsum_ratio = np.cumsum(temp_pca.explained_variance_ratio_)
                self.n_components = np.argmax(cumsum_ratio >= self.variance_threshold) + 1
            
            self.reducer = PCA(n_components=self.n_components)
            self.reducer.fit(X)
            
            self.logger.info(f"PCA fitted with {self.n_components} components")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using dimensionality reduction."""
        X_reduced = self.reducer.transform(X)
        
        # Create column names
        if self.method == 'pca':
            columns = [f'PC{i+1}' for i in range(X_reduced.shape[1])]
        else:
            columns = [f'Component_{i+1}' for i in range(X_reduced.shape[1])]
        
        return pd.DataFrame(X_reduced, columns=columns, index=X.index)
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance/loadings."""
        if not self.is_fitted:
            return None
        
        if self.method == 'pca' and hasattr(self.reducer, 'components_'):
            components_df = pd.DataFrame(
                self.reducer.components_.T,
                columns=[f'PC{i+1}' for i in range(self.reducer.n_components_)],
                index=range(self.reducer.n_features_in_)
            )
            return components_df
        
        return None


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select the most relevant features for anomaly detection."""
    
    def __init__(self, method: str = 'mutual_info', k: int = 20, threshold: Optional[float] = None):
        self.method = method
        self.k = k
        self.threshold = threshold
        self.selector = None
        self.selected_features = []
        self.is_fitted = False
        self.logger = model_logger
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit the feature selector."""
        if y is None:
            # For unsupervised feature selection, use variance
            from sklearn.feature_selection import VarianceThreshold
            threshold = self.threshold or 0.01
            self.selector = VarianceThreshold(threshold=threshold)
            self.selector.fit(X)
            
            # Get selected feature names
            selected_idx = self.selector.get_support()
            self.selected_features = X.columns[selected_idx].tolist()
        else:
            # For supervised feature selection
            if self.method == 'mutual_info':
                self.selector = SelectKBest(score_func=mutual_info_classif, k=self.k)
            else:
                self.selector = SelectKBest(score_func=f_classif, k=self.k)
            
            self.selector.fit(X, y)
            
            # Get selected feature names
            selected_idx = self.selector.get_support()
            self.selected_features = X.columns[selected_idx].tolist()
        
        self.is_fitted = True
        self.logger.info(f"Selected {len(self.selected_features)} features using {self.method}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by selecting features."""
        return X[self.selected_features]
    
    def get_feature_scores(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        if not self.is_fitted or not hasattr(self.selector, 'scores_'):
            return None
        
        if hasattr(self.selector, 'get_support'):
            selected_idx = self.selector.get_support()
            scores = self.selector.scores_[selected_idx]
            return dict(zip(self.selected_features, scores))
        
        return None 