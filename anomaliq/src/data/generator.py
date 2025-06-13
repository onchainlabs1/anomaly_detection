"""
Synthetic data generator for the Anomaliq system.
Generates realistic credit card transaction data for testing and demonstration.
Mimics the structure of the Credit Card Fraud dataset from Kaggle.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import warnings

from src.utils import get_settings, data_logger, ensure_dir


class CreditCardDataGenerator:
    """Generates synthetic credit card transaction data."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.settings = get_settings()
        self.logger = data_logger
        np.random.seed(random_state)
    
    def generate_normal_transactions(self, n_samples: int) -> pd.DataFrame:
        """Generate normal (non-fraudulent) transactions."""
        np.random.seed(self.random_state)
        
        # Generate PCA-like features V1-V28 (anonymized features)
        # These would typically be the result of PCA transformation
        # Normal transactions cluster around zero with some variation
        data = {}
        
        for i in range(1, 29):  # V1 to V28
            if i <= 10:
                # First 10 features have wider distribution
                data[f'V{i}'] = np.random.normal(0, 2.0, n_samples)
            elif i <= 20:
                # Middle features are more concentrated
                data[f'V{i}'] = np.random.normal(0, 1.5, n_samples)
            else:
                # Last features are tightly concentrated
                data[f'V{i}'] = np.random.normal(0, 1.0, n_samples)
        
        # Time feature (seconds elapsed between transactions)
        # Simulate transactions throughout the day
        data['Time'] = np.random.exponential(300, n_samples).cumsum()  # Average 5 minutes between transactions
        data['Time'] = data['Time'] % (24 * 3600)  # Wrap around 24 hours
        
        # Amount feature (transaction amounts)
        # Most transactions are small, few are large
        amounts = np.random.lognormal(mean=3.0, sigma=1.5, size=n_samples)
        data['Amount'] = np.clip(amounts, 0.01, 10000)  # Clip extreme values
        
        # Class (0 for normal)
        data['Class'] = np.zeros(n_samples, dtype=int)
        
        return pd.DataFrame(data)
    
    def generate_fraudulent_transactions(self, n_samples: int) -> pd.DataFrame:
        """Generate fraudulent transactions with different patterns."""
        np.random.seed(self.random_state + 999)  # Different seed for fraud
        
        data = {}
        
        # Fraudulent transactions have different feature distributions
        for i in range(1, 29):  # V1 to V28
            if i in [1, 2, 3, 4, 14, 17, 18]:  # Key discriminative features
                # These features show clear separation for fraud
                data[f'V{i}'] = np.random.normal(3.0, 2.0, n_samples)  # Shifted distribution
            elif i in [10, 11, 12, 16, 19]:
                # These features show moderate separation
                data[f'V{i}'] = np.random.normal(-2.0, 1.5, n_samples)  # Shifted negative
            else:
                # Other features similar to normal but with more variance
                data[f'V{i}'] = np.random.normal(0, 3.0, n_samples)
        
        # Fraudulent transactions often occur at unusual times
        # More likely during night hours or in bursts
        night_transactions = np.random.choice([True, False], size=n_samples, p=[0.6, 0.4])
        
        # Night time transactions (22:00 - 06:00)
        night_times = np.random.uniform(22*3600, 30*3600, size=np.sum(night_transactions)) % (24*3600)
        day_times = np.random.uniform(6*3600, 22*3600, size=n_samples - np.sum(night_transactions))
        
        all_times = np.concatenate([night_times, day_times])
        np.random.shuffle(all_times)
        data['Time'] = all_times
        
        # Fraudulent amounts tend to be either very small (testing) or specific amounts
        fraud_types = np.random.choice(['small', 'medium', 'large'], size=n_samples, p=[0.4, 0.4, 0.2])
        
        amounts = []
        for fraud_type in fraud_types:
            if fraud_type == 'small':
                amounts.append(np.random.uniform(0.01, 10))  # Small test amounts
            elif fraud_type == 'medium':
                amounts.append(np.random.uniform(50, 500))   # Medium amounts
            else:
                amounts.append(np.random.uniform(1000, 5000))  # Large amounts
        
        data['Amount'] = amounts
        
        # Class (1 for fraud)
        data['Class'] = np.ones(n_samples, dtype=int)
        
        return pd.DataFrame(data)
    
    def generate_dataset(
        self, 
        n_samples: int = 10000,
        fraud_ratio: float = 0.001,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Generate complete dataset with normal and fraudulent transactions."""
        self.logger.info(f"Generating synthetic dataset with {n_samples} samples")
        
        # Calculate number of fraud samples
        n_fraud = int(n_samples * fraud_ratio)
        n_normal = n_samples - n_fraud
        
        self.logger.info(f"Normal transactions: {n_normal}, Fraudulent: {n_fraud}")
        
        # Generate normal and fraudulent transactions
        normal_df = self.generate_normal_transactions(n_normal)
        fraud_df = self.generate_fraudulent_transactions(n_fraud)
        
        # Combine and shuffle
        df = pd.concat([normal_df, fraud_df], ignore_index=True)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Add some realistic data variations
        df = self._add_data_variations(df)
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            df.to_csv(save_path, index=False)
            self.logger.info(f"Dataset saved to: {save_path}")
        
        self.logger.info("Dataset generation completed")
        return df
    
    def _add_data_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic variations to the data."""
        df = df.copy()
        
        # Add some missing values (very few in credit card data)
        missing_cols = ['V1', 'V2', 'Amount']
        for col in missing_cols:
            missing_indices = np.random.choice(
                len(df), 
                size=int(len(df) * 0.001),  # 0.1% missing
                replace=False
            )
            df.loc[missing_indices, col] = np.nan
        
        # Add some outliers to normal transactions
        outlier_indices = np.random.choice(
            df[df['Class'] == 0].index,
            size=int(len(df) * 0.005),  # 0.5% outliers
            replace=False
        )
        
        # Make some V features extreme for outliers
        for idx in outlier_indices:
            feature = np.random.choice(['V1', 'V2', 'V3', 'V14'])
            df.loc[idx, feature] = np.random.choice([-10, 10]) * np.random.uniform(1, 3)
        
        return df
    
    def generate_drift_data(
        self,
        base_data: pd.DataFrame,
        drift_magnitude: float = 0.3,
        n_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate data with concept drift for testing monitoring."""
        if n_samples is None:
            n_samples = len(base_data)
        
        self.logger.info(f"Generating drift data with magnitude {drift_magnitude}")
        
        # Start with base data distribution
        df = self.generate_dataset(n_samples, fraud_ratio=0.001)
        
        # Apply drift to some features
        drift_features = ['V1', 'V2', 'V3', 'V4', 'Amount']
        
        for feature in drift_features:
            if feature == 'Amount':
                # Shift transaction amounts (inflation effect)
                df[feature] = df[feature] * (1 + drift_magnitude)
            else:
                # Shift feature distributions
                shift = drift_magnitude * df[feature].std()
                df[feature] = df[feature] + shift
        
        # Change fraud patterns slightly
        fraud_mask = df['Class'] == 1
        for feature in ['V14', 'V17', 'V18']:
            df.loc[fraud_mask, feature] = df.loc[fraud_mask, feature] * (1 + drift_magnitude * 0.5)
        
        return df
    
    def create_sample_datasets(self, output_dir: str = "./data/"):
        """Create sample datasets for development and testing."""
        output_path = Path(output_dir)
        ensure_dir(output_path)
        
        self.logger.info("Creating sample datasets...")
        
        # Training dataset (larger)
        train_data = self.generate_dataset(
            n_samples=50000,
            fraud_ratio=0.002,
            save_path=output_path / "training_data.csv"
        )
        
        # Test dataset
        test_data = self.generate_dataset(
            n_samples=10000,
            fraud_ratio=0.002,
            save_path=output_path / "test_data.csv"
        )
        
        # Reference data for drift monitoring (subset of training)
        reference_data = train_data.sample(n=5000, random_state=42)
        reference_data.to_csv(output_path / "reference_data.csv", index=False)
        
        # Live data with slight drift
        live_data = self.generate_drift_data(
            reference_data,
            drift_magnitude=0.1,
            n_samples=5000
        )
        live_data.to_csv(output_path / "live_data.csv", index=False)
        
        # Inference data (no labels)
        inference_data = test_data.drop('Class', axis=1).sample(n=1000, random_state=42)
        inference_data.to_csv(output_path / "inference_data.csv", index=False)
        
        self.logger.info("Sample datasets created successfully")
        
        return {
            "training": train_data,
            "test": test_data,
            "reference": reference_data,
            "live": live_data,
            "inference": inference_data
        }


def generate_sample_record() -> dict:
    """Generate a single sample record for API testing."""
    generator = CreditCardDataGenerator()
    
    # Generate one normal transaction
    sample_df = generator.generate_normal_transactions(1)
    
    # Convert to dictionary (excluding Class for inference)
    record = sample_df.drop('Class', axis=1).iloc[0].to_dict()
    
    return record


def main():
    """Generate sample datasets when run as script."""
    generator = CreditCardDataGenerator()
    datasets = generator.create_sample_datasets()
    
    print("\nDataset Summary:")
    for name, df in datasets.items():
        print(f"{name.capitalize()}: {len(df)} samples")
        if 'Class' in df.columns:
            fraud_count = (df['Class'] == 1).sum()
            print(f"  - Fraud rate: {fraud_count/len(df)*100:.3f}%")


if __name__ == "__main__":
    main() 