"""
Unit tests for model components of the Anomaliq system.
Tests model training, prediction, and evaluation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.ensemble import IsolationForest

from src.models.train_model import AnomalyModelTrainer
from src.models.predictor import AnomalyPredictor
from src.data import CreditCardDataGenerator


class TestAnomalyModelTrainer:
    """Test cases for AnomalyModelTrainer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        generator = CreditCardDataGenerator()
        return generator.generate_dataset(n_samples=1000, fraud_ratio=0.01)
    
    @pytest.fixture
    def trainer(self):
        """Create trainer instance."""
        return AnomalyModelTrainer()
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initializes correctly."""
        assert trainer is not None
        assert hasattr(trainer, 'settings')
        assert hasattr(trainer, 'model_config')
        assert hasattr(trainer, 'logger')
    
    def test_prepare_features(self, trainer, sample_data):
        """Test feature preparation."""
        X = sample_data.drop('Class', axis=1)
        y = sample_data['Class']
        
        X_transformed, feature_pipeline = trainer.prepare_features(X, y)
        
        assert X_transformed is not None
        assert feature_pipeline is not None
        assert len(X_transformed) == len(X)
        assert feature_pipeline.is_fitted
    
    def test_train_isolation_forest(self, trainer, sample_data):
        """Test Isolation Forest training."""
        X = sample_data.drop('Class', axis=1)
        y = sample_data['Class']
        
        X_transformed, _ = trainer.prepare_features(X, y)
        model = trainer.train_isolation_forest(X_transformed, y)
        
        assert isinstance(model, IsolationForest)
        assert hasattr(model, 'decision_function')
        assert hasattr(model, 'predict')
    
    def test_evaluate_model(self, trainer, sample_data):
        """Test model evaluation."""
        X = sample_data.drop('Class', axis=1)
        y = sample_data['Class']
        
        X_transformed, _ = trainer.prepare_features(X, y)
        model = trainer.train_isolation_forest(X_transformed, y)
        
        metrics = trainer.evaluate_model(model, X_transformed, y)
        
        assert isinstance(metrics, dict)
        assert 'anomaly_score_mean' in metrics
        assert 'predicted_anomaly_rate' in metrics
        assert 'n_samples' in metrics
        assert 'n_features' in metrics
        
        # If labels are provided, should have additional metrics
        if y is not None:
            assert 'auc_roc' in metrics or metrics['auc_roc'] is None


class TestAnomalyPredictor:
    """Test cases for AnomalyPredictor."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing."""
        model = MagicMock()
        model.decision_function.return_value = np.array([-0.3])
        model.predict.return_value = np.array([-1])  # Anomaly
        return model
    
    @pytest.fixture
    def predictor(self, mock_model):
        """Create predictor with mock model."""
        predictor = AnomalyPredictor()
        predictor.model = mock_model
        return predictor
    
    def test_predictor_initialization(self):
        """Test predictor initializes correctly."""
        predictor = AnomalyPredictor()
        assert predictor is not None
        assert hasattr(predictor, 'settings')
        assert hasattr(predictor, 'model_config')
        assert hasattr(predictor, 'logger')
    
    def test_predict_single_dict_input(self, predictor):
        """Test single prediction with dictionary input."""
        record = {
            'V1': 1.0, 'V2': 2.0, 'V3': 3.0, 'V4': 4.0,
            'Amount': 100.0
        }
        
        result = predictor.predict_single(record, explain=False)
        
        assert isinstance(result, dict)
        assert 'anomaly_score' in result
        assert 'is_anomaly' in result
        assert 'confidence' in result
        assert 'threshold' in result
        
        assert isinstance(result['anomaly_score'], float)
        assert isinstance(result['is_anomaly'], bool)
        assert isinstance(result['confidence'], float)
    
    def test_predict_single_dataframe_input(self, predictor):
        """Test single prediction with DataFrame input."""
        df = pd.DataFrame({
            'V1': [1.0], 'V2': [2.0], 'V3': [3.0], 'V4': [4.0],
            'Amount': [100.0]
        })
        
        result = predictor.predict_single(df, explain=False)
        
        assert isinstance(result, dict)
        assert 'anomaly_score' in result
        assert 'is_anomaly' in result
    
    def test_predict_single_invalid_input(self, predictor):
        """Test prediction with invalid input type."""
        with pytest.raises(ValueError):
            predictor.predict_single("invalid_input")
    
    def test_predict_batch(self, predictor):
        """Test batch prediction."""
        df = pd.DataFrame({
            'V1': [1.0, 2.0, 3.0], 
            'V2': [2.0, 3.0, 4.0], 
            'V3': [3.0, 4.0, 5.0],
            'Amount': [100.0, 200.0, 300.0]
        })
        
        predictor.model.decision_function.return_value = np.array([-0.3, 0.1, -0.5])
        
        scores = predictor.predict_batch(df, explain=False, return_details=False)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(df)
    
    def test_predict_batch_with_details(self, predictor):
        """Test batch prediction with details."""
        df = pd.DataFrame({
            'V1': [1.0, 2.0], 
            'V2': [2.0, 3.0],
            'Amount': [100.0, 200.0]
        })
        
        predictor.model.decision_function.return_value = np.array([-0.3, 0.1])
        
        result = predictor.predict_batch(df, explain=False, return_details=True)
        
        assert isinstance(result, dict)
        assert 'anomaly_scores' in result
        assert 'is_anomaly' in result
        assert 'n_anomalies' in result
        assert 'anomaly_rate' in result
    
    def test_get_model_info(self, predictor):
        """Test getting model information."""
        info = predictor.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_type' in info
        assert 'has_feature_pipeline' in info
        assert 'has_shap_explainer' in info
        assert 'threshold' in info
    
    def test_update_threshold(self, predictor):
        """Test updating anomaly threshold."""
        new_threshold = -0.2
        old_threshold = predictor.settings.model_threshold
        
        predictor.update_threshold(new_threshold)
        
        assert predictor.settings.model_threshold == new_threshold
        assert predictor.settings.model_threshold != old_threshold


class TestIntegration:
    """Integration tests for model components."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for integration testing."""
        generator = CreditCardDataGenerator()
        return generator.generate_dataset(n_samples=500, fraud_ratio=0.02)
    
    def test_full_training_pipeline(self, sample_data, tmp_path):
        """Test complete training pipeline."""
        # Save sample data
        data_path = tmp_path / "sample_data.csv"
        sample_data.to_csv(data_path, index=False)
        
        # Initialize trainer
        trainer = AnomalyModelTrainer()
        
        # Mock MLflow to avoid actual tracking
        with patch('mlflow.start_run'), \
             patch('mlflow.log_param'), \
             patch('mlflow.log_metric'), \
             patch('mlflow.sklearn.log_model'):
            
            # Train model (this would normally use MLflow)
            try:
                X = sample_data.drop('Class', axis=1)
                y = sample_data['Class']
                
                # Prepare features
                X_transformed, feature_pipeline = trainer.prepare_features(X, y)
                
                # Train model
                model = trainer.train_isolation_forest(X_transformed, y)
                
                # Evaluate model
                metrics = trainer.evaluate_model(model, X_transformed, y)
                
                # Verify results
                assert model is not None
                assert feature_pipeline is not None
                assert isinstance(metrics, dict)
                assert len(metrics) > 0
                
            except Exception as e:
                # Expected to fail without proper MLflow setup
                assert "MLflow" in str(e) or "experiment" in str(e).lower()
    
    def test_training_to_prediction_pipeline(self, sample_data):
        """Test pipeline from training to prediction."""
        # Prepare data
        X = sample_data.drop('Class', axis=1)
        y = sample_data['Class']
        
        # Train model
        trainer = AnomalyModelTrainer()
        X_transformed, feature_pipeline = trainer.prepare_features(X, y)
        model = trainer.train_isolation_forest(X_transformed, y)
        
        # Create predictor with trained model
        predictor = AnomalyPredictor()
        predictor.model = model
        predictor.feature_pipeline = feature_pipeline
        
        # Test prediction
        sample_record = X.iloc[0].to_dict()
        result = predictor.predict_single(sample_record, explain=False)
        
        assert isinstance(result, dict)
        assert 'anomaly_score' in result
        assert 'is_anomaly' in result
        assert isinstance(result['anomaly_score'], float)
        assert isinstance(result['is_anomaly'], bool)


if __name__ == "__main__":
    pytest.main([__file__]) 