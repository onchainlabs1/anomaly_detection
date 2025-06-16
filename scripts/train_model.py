import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri("./mlruns")

# Load and prepare data
print("Loading data...")
data = pd.read_csv('anomaliq/data/reference_data.csv')
feature_columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
X = data[feature_columns]

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
print("Training model...")
model = IsolationForest(
    contamination=0.1,
    n_estimators=100,
    max_samples='auto',
    max_features=1.0,
    bootstrap=False,
    random_state=42
)
model.fit(X_scaled)

# Create MLflow experiment if it doesn't exist
experiment_name = "credit_card_fraud_detection"
client = MlflowClient()
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Log model with MLflow
print("Logging model to MLflow...")
mlflow.set_experiment(experiment_name)
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("contamination", 0.1)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_features", 1.0)
    mlflow.log_param("bootstrap", False)
    
    # Log model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="isolation_forest_anomaly_detector"
    )
    
    # Log the scaler as well
    mlflow.sklearn.log_model(
        sk_model=scaler,
        artifact_path="scaler"
    )

print("Model training and registration complete!") 