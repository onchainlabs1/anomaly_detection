#!/usr/bin/env python3
"""
Model training script for the Anomaliq system.
Trains anomaly detection models with MLflow experiment tracking.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from loguru import logger

from src.data import DataLoader, DataValidator
from src.features import FeaturePipeline
from src.utils import (
    get_settings, get_model_config, model_logger, 
    log_model_training_start, log_model_training_complete,
    ensure_dir, get_timestamp
)


class AnomalyModelTrainer:
    """Trains and evaluates anomaly detection models."""
    
    def __init__(self):
        self.settings = get_settings()
        self.model_config = get_model_config()
        self.logger = model_logger
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.settings.mlflow_experiment_name)
    
    def load_and_prepare_data(
        self,
        data_path: Optional[Union[str, Path]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and prepare data for training."""
        self.logger.info("Loading and preparing training data")
        
        data_loader = DataLoader()
        
        # Use creditcard.csv as the training data source
        df = data_loader.load_training_data(file_path="anomaliq/src/data/creditcard.csv")
        
        # Prepare features and target
        X, y = data_loader.prepare_features_target(df)
        
        return X, y
    
    def prepare_features(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, FeaturePipeline]:
        """Prepare features using feature engineering pipeline."""
        self.logger.info("Preparing features")
        
        # Create feature pipeline
        feature_pipeline = FeaturePipeline(
            include_time_features=True,
            scaler_type=self.model_config.scaler_type,
            handle_outliers=True
        )
        
        # Fit and transform features
        X_transformed = feature_pipeline.fit_transform(X, y)
        
        self.logger.info(f"Features transformed: {X_transformed.shape[1]} features")
        return X_transformed, feature_pipeline
    
    def train_isolation_forest(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> IsolationForest:
        """Train Isolation Forest model."""
        self.logger.info("Training Isolation Forest model")
        
        # Initialize model with configuration
        model = IsolationForest(
            contamination=self.model_config.contamination,
            n_estimators=self.model_config.n_estimators,
            max_samples=self.model_config.max_samples,
            random_state=self.model_config.random_state,
            n_jobs=-1
        )
        
        # Fit model
        model.fit(X)
        
        self.logger.info("Isolation Forest model trained successfully")
        return model
    
    def evaluate_model(
        self, 
        model: IsolationForest, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Evaluate trained model."""
        self.logger.info("Evaluating model")
        
        # Get anomaly scores
        anomaly_scores = model.decision_function(X)
        predictions = model.predict(X)  # -1 for anomaly, 1 for normal
        
        # Convert predictions to binary (1 for anomaly, 0 for normal) 
        anomaly_predictions = (predictions == -1).astype(int)
        
        metrics = {
            'anomaly_score_mean': float(np.mean(anomaly_scores)),
            'anomaly_score_std': float(np.std(anomaly_scores)),
            'anomaly_score_min': float(np.min(anomaly_scores)),
            'anomaly_score_max': float(np.max(anomaly_scores)),
            'predicted_anomaly_rate': float(np.mean(anomaly_predictions)),
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        # If we have true labels, calculate additional metrics
        if y is not None:
            # Convert target to binary (assuming 1 is anomaly, 0 is normal)
            true_anomalies = y.astype(int)
            
            try:
                auc_score = roc_auc_score(true_anomalies, -anomaly_scores)  # Negative because lower scores = more anomalous
                metrics['auc_roc'] = float(auc_score)
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC: {e}")
                metrics['auc_roc'] = None
            
            # Classification metrics using default threshold
            cm = confusion_matrix(true_anomalies, anomaly_predictions)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics.update({
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn)
                })
        
        self.logger.info(f"Model evaluation completed: {len(metrics)} metrics calculated")
        return metrics
    
    def train_and_evaluate(
        self, 
        data_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Complete training and evaluation pipeline."""
        
        # Set experiment name
        if experiment_name:
            mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            self.logger.info(f"Started MLflow run: {run_id}")
            
            # Log training start
            log_model_training_start(
                experiment_name or self.settings.mlflow_experiment_name,
                "IsolationForest"
            )
            
            try:
                # Load and prepare data
                X, y = self.load_and_prepare_data(data_path)
                
                # Split data if we have labels
                if y is not None:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=self.model_config.test_size,
                        random_state=self.model_config.random_state,
                        stratify=y
                    )
                else:
                    X_train, X_test = train_test_split(
                        X, 
                        test_size=self.model_config.test_size,
                        random_state=self.model_config.random_state
                    )
                    y_train, y_test = None, None
                
                # Prepare features
                X_train_processed, feature_pipeline = self.prepare_features(X_train, y_train)
                X_test_processed = feature_pipeline.transform(X_test)
                
                # Log data information
                mlflow.log_param("n_train_samples", len(X_train))
                mlflow.log_param("n_test_samples", len(X_test))
                mlflow.log_param("n_features_original", X.shape[1])
                mlflow.log_param("n_features_processed", X_train_processed.shape[1])
                
                # Log model parameters
                mlflow.log_param("contamination", self.model_config.contamination)
                mlflow.log_param("n_estimators", self.model_config.n_estimators)
                mlflow.log_param("max_samples", self.model_config.max_samples)
                mlflow.log_param("scaler_type", self.model_config.scaler_type)
                
                # Train model
                model = self.train_isolation_forest(X_train_processed, y_train)
                
                # Evaluate on training set
                train_metrics = self.evaluate_model(model, X_train_processed, y_train)
                for key, value in train_metrics.items():
                    if value is not None:
                        mlflow.log_metric(f"train_{key}", value)
                
                # Evaluate on test set
                test_metrics = self.evaluate_model(model, X_test_processed, y_test)
                for key, value in test_metrics.items():
                    if value is not None:
                        mlflow.log_metric(f"test_{key}", value)
                
                # Save model and pipeline
                model_name = model_name or f"isolation_forest_{get_timestamp()}"
                
                # Log model with MLflow
                mlflow.sklearn.log_model(
                    model, 
                    "model",
                    registered_model_name=model_name
                )
                
                # Save feature pipeline separately
                pipeline_path = f"models/{model_name}_pipeline.joblib"
                ensure_dir(Path(pipeline_path).parent)
                feature_pipeline.save(pipeline_path)
                mlflow.log_artifact(pipeline_path, "preprocessing")
                
                # Log completion
                final_metrics = {**train_metrics, **{f"test_{k}": v for k, v in test_metrics.items()}}
                log_model_training_complete(
                    experiment_name or self.settings.mlflow_experiment_name,
                    "IsolationForest",
                    final_metrics
                )
                
                self.logger.info(f"Training completed successfully. Run ID: {run_id}")
                
                return {
                    "run_id": run_id,
                    "model_name": model_name,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "model": model,
                    "feature_pipeline": feature_pipeline
                }
                
            except Exception as e:
                self.logger.error(f"Training failed: {str(e)}")
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
                raise


def main():
    """Main function for command line training."""
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument("--data-path", type=str, help="Path to training data CSV file")
    parser.add_argument("--experiment-name", type=str, help="MLflow experiment name")
    parser.add_argument("--model-name", type=str, help="Name for the trained model")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = AnomalyModelTrainer()
    
    # Train model
    try:
        results = trainer.train_and_evaluate(
            data_path=args.data_path,
            experiment_name=args.experiment_name,
            model_name=args.model_name
        )
        
        print(f"\n✅ Training completed successfully!")
        print(f"Run ID: {results['run_id']}")
        print(f"Model Name: {results['model_name']}")
        print("\nTest Metrics:")
        for key, value in results['test_metrics'].items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 