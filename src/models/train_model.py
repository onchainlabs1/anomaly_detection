#!/usr/bin/env python3
"""
Model training script for the Anomaliq system.
Trains anomaly detection models with MLflow experiment tracking.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

from src.data import DataLoader, DataValidator
from src.features import FeaturePipeline
from src.utils import (
    get_settings, get_model_config, model_logger, 
    log_model_training_start, log_model_training_complete,
    ensure_dir, get_timestamp
)


class AnomalyModelTrainer:
    """Trains and evaluates anomaly detection models."""
    
    def __init__(self, experiment_name: str = None):
        """Initialize trainer with settings and logging."""
        self.settings = get_settings()
        self.logger = model_logger
        self.experiment_name = experiment_name or self.settings.mlflow_experiment_name
        self.model_config = get_model_config()
        
        # Set up MLflow tracking
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        ensure_dir(self.settings.mlflow_tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment = mlflow.create_experiment(self.experiment_name)
        except Exception:
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
    
    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Load and prepare data for training."""
        self.logger.info("Loading and preparing training data")
        
        # Initialize data loader and validator
        data_loader = DataLoader()
        data_validator = DataValidator()
        
        # Load and validate data
        df = data_loader.load_csv(data_path)
        data_validator.validate_for_training(df)
        
        # Prepare features and target
        X, y = data_loader.prepare_features_target(df)
        
        return X, y
    
    def train_model(self, X: pd.DataFrame) -> IsolationForest:
        """Train the anomaly detection model."""
        self.logger.info("Training model")
        
        # Initialize and train model
        model = IsolationForest(
            n_estimators=self.model_config.get("n_estimators", 100),
            max_samples=self.model_config.get("max_samples", "auto"),
            contamination=self.model_config.get("contamination", "auto"),
            max_features=self.model_config.get("max_features", 1.0),
            bootstrap=self.model_config.get("bootstrap", False),
            n_jobs=self.model_config.get("n_jobs", -1),
            random_state=self.model_config.get("random_state", 42),
            verbose=0
        )
        
        model.fit(X)
        return model
    
    def evaluate_model(
        self, 
        model: IsolationForest, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Evaluate the trained model."""
        self.logger.info("Evaluating model")
        
        # Get model predictions
        y_pred = model.predict(X)
        y_score = model.score_samples(X)
        
        # Convert predictions to binary format (1: normal, 0: anomaly)
        y_pred_binary = np.where(y_pred == 1, 0, 1)
        
        # Calculate metrics
        metrics = {
            "num_samples": len(X),
            "num_features": X.shape[1],
            "anomaly_ratio": np.mean(y_pred_binary)
        }
        
        # If true labels are available, calculate additional metrics
        if y is not None:
            metrics.update({
                "roc_auc_score": roc_auc_score(y, y_score),
                "classification_report": classification_report(y, y_pred_binary),
                "confusion_matrix": confusion_matrix(y, y_pred_binary).tolist()
            })
        
        return metrics
    
    def save_model(self, model: IsolationForest, metrics: Dict[str, Any]) -> None:
        """Save the trained model and metrics."""
        self.logger.info("Saving model")
        
        # Save model with MLflow
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=self.settings.mlflow_model_name
        )
        
        # Log metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)
            else:
                mlflow.log_param(key, str(value))
    
    def train_and_evaluate(self, data_path: str) -> None:
        """Train and evaluate the model."""
        try:
            # Start MLflow run
            with mlflow.start_run(experiment_id=self.experiment) as run:
                self.logger.info(f"Started MLflow run: {run.info.run_id}")
                
                # Log training start
                log_model_training_start(
                    experiment_name=self.experiment_name,
                    model_type="IsolationForest"
                )
                
                # Load and prepare data
                X, y = self.load_and_prepare_data(data_path)
                
                # Train model
                model = self.train_model(X)
                
                # Evaluate model
                metrics = self.evaluate_model(model, X, y)
                
                # Save model and metrics
                self.save_model(model, metrics)
                
                # Log training completion
                log_model_training_complete(metrics)
                
                self.logger.info("Model training completed successfully")
                
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise


def main():
    """Main function to run model training."""
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="MLflow experiment name"
    )
    
    args = parser.parse_args()
    
    trainer = AnomalyModelTrainer(experiment_name=args.experiment_name)
    trainer.train_and_evaluate(args.data_path)


if __name__ == "__main__":
    main() 