# Anomaly Detection System

A real-time anomaly detection system for credit card transactions using FastAPI and Streamlit.

## Features

- Real-time anomaly detection using Isolation Forest
- FastAPI backend with JWT authentication
- Interactive Streamlit dashboard
- MLflow model management
- Synthetic data generation for testing

## Project Structure

```
anomaliq/
├── src/
│   ├── api/           # FastAPI application
│   ├── dashboard/     # Streamlit dashboard
│   ├── data/         # Data generation and processing
│   ├── models/       # ML models and predictors
│   └── utils/        # Shared utilities
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/onchainlabs1/anomaly_detection.git
cd anomaly_detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the API:
```bash
PYTHONPATH=/path/to/project python -m uvicorn anomaliq.src.api.main:app --host 0.0.0.0 --port 8000
```

4. Start the dashboard:
```bash
PYTHONPATH=/path/to/project streamlit run anomaliq/src/dashboard/app.py
```

## Usage

1. Access the dashboard at http://localhost:8501
2. Login with default credentials (admin/admin)
3. Generate synthetic transactions
4. View anomaly predictions in real-time

## API Documentation

The API documentation is available at http://localhost:8000/docs when the server is running.

## License

MIT License 