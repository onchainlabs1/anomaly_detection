# Anomaly Detection System

A real-time anomaly detection system for credit card transactions using FastAPI, MLflow, and Streamlit.

## Features

- Real-time anomaly detection using Isolation Forest
- FastAPI backend with JWT authentication
- Interactive Streamlit dashboard
- MLflow model management and experiment tracking
- Data drift detection and monitoring
- Synthetic data generation for testing

## Project Structure

```
anomaliq/
├── src/
│   ├── api/           # FastAPI application
│   ├── dashboard/     # Streamlit dashboard
│   ├── data/         # Data loading and validation
│   ├── models/       # ML models and predictors
│   ├── monitoring/   # Drift detection and monitoring
│   └── utils/        # Shared utilities
├── scripts/          # Training and utility scripts
└── mlruns/          # MLflow tracking directory
```

## Requirements

- Python 3.11+
- MLflow
- FastAPI
- Streamlit
- scikit-learn
- evidently

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/anomaly.git
cd anomaly
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start MLflow server:
```bash
mlflow server --host 0.0.0.0 --port 5001
```

4. Train the model:
```bash
PYTHONPATH=/path/to/anomaly/anomaliq python scripts/train_model.py
```

5. Start the API:
```bash
PYTHONPATH=/path/to/anomaly/anomaliq uvicorn anomaliq.src.api.main:app --host 0.0.0.0 --port 8000
```

6. Start the dashboard:
```bash
PYTHONPATH=/path/to/anomaly/anomaliq streamlit run anomaliq/src/dashboard/app.py
```

## Usage

1. Access the MLflow UI at http://localhost:5001
2. Access the dashboard at http://localhost:8501
3. Login with default credentials (admin/admin)
4. Upload transaction data or use synthetic data
5. View anomaly predictions and drift detection in real-time

## API Documentation

The API documentation is available at http://localhost:8000/docs when the server is running.

## Environment Variables

The system uses the following environment variables (configured in `config.env`):

- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `MLFLOW_TRACKING_URI`: MLflow tracking URI
- `MLFLOW_EXPERIMENT_NAME`: MLflow experiment name
- `MLFLOW_MODEL_NAME`: MLflow model name
- `STREAMLIT_HOST`: Streamlit host
- `STREAMLIT_PORT`: Streamlit port

## License

MIT License 