# Anomaliq - Production-Ready Anomaly Detection System

A comprehensive MLOps solution for detecting anomalous financial and operational records using unsupervised learning techniques. Built with FastAPI, Streamlit, MLflow, and SHAP for explainable AI.

## 🚀 Features

- **FastAPI Backend**: RESTful API with JWT authentication for anomaly prediction
- **Streamlit Dashboard**: Interactive visualization with filters and SHAP explanations
- **MLflow Integration**: Experiment tracking, model versioning, and metrics logging
- **SHAP Explainability**: Understanding why certain records are flagged as anomalous
- **Data Drift Monitoring**: Automated detection of distribution shifts in production data
- **Clean Architecture**: Modular design with separation of concerns
- **Production Ready**: Comprehensive logging, testing, and monitoring

## 📁 Project Structure

```
anomaliq/
├── src/
│   ├── api/                    # FastAPI backend
│   │   ├── main.py            # API entry point
│   │   ├── routes.py          # API routes
│   │   └── auth.py            # JWT authentication
│   ├── dashboard/              # Streamlit frontend
│   │   ├── app.py             # Dashboard entry point
│   │   └── views.py           # Dashboard views and components
│   ├── models/                 # Model training and inference
│   │   ├── train_model.py     # Model training with MLflow
│   │   └── predictor.py       # Model inference and prediction
│   ├── features/               # Feature engineering
│   │   ├── preprocessing.py   # Data preprocessing pipeline
│   │   └── transformations.py # Feature transformations
│   ├── data/                   # Data handling
│   │   ├── loader.py          # Data loading utilities
│   │   └── generator.py       # Synthetic data generation
│   ├── utils/                  # Utilities and configuration
│   │   ├── config.py          # Configuration management
│   │   ├── logging.py         # Logging setup
│   │   └── helpers.py         # Helper functions
│   └── monitoring/             # Monitoring and alerting
│       ├── drift_detection.py # Data drift monitoring
│       └── metrics.py         # Performance metrics
├── notebooks/                  # Jupyter notebooks for EDA
├── tests/                      # Unit and integration tests
├── mlruns/                     # MLflow experiment runs
├── run_with_landing.py         # Unified entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd anomaliq
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory with the following variables:
   ```env
   # API Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   SECRET_KEY=your-super-secret-jwt-key-change-in-production
   
   # MLflow Configuration
   MLFLOW_TRACKING_URI=./mlruns
   MLFLOW_EXPERIMENT_NAME=anomaly_detection
   
   # Model Configuration
   MODEL_THRESHOLD=0.1
   FEATURE_COLUMNS=V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount
   ```

## 🚀 Quick Start

### Using the Unified CLI

The project includes a unified entry point script for easy management:

```bash
# Train the model
python run_with_landing.py train

# Start the API server
python run_with_landing.py api

# Start the Streamlit dashboard
python run_with_landing.py dashboard

# Start both API and dashboard
python run_with_landing.py full

# Launch MLflow UI
python run_with_landing.py mlflow

# Run tests
python run_with_landing.py test

# Run monitoring
python run_with_landing.py monitor
```

### Manual Commands

If you prefer running components individually:

```bash
# Train the model
python -m src.models.train_model

# Start API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start Streamlit dashboard
streamlit run src/dashboard/app.py

# Launch MLflow UI
mlflow ui --backend-store-uri ./mlruns
```

## 📊 Usage

### API Endpoints

The FastAPI backend provides the following endpoints:

- `POST /auth/token` - Get JWT access token
- `POST /predict_anomaly` - Predict anomaly score for a record
- `GET /health` - Health check endpoint
- `GET /metrics` - Model performance metrics

#### Example API Usage

```python
import requests

# Get access token
auth_response = requests.post("http://localhost:8000/auth/token", {
    "username": "admin",
    "password": "admin"
})
token = auth_response.json()["access_token"]

# Predict anomaly
headers = {"Authorization": f"Bearer {token}"}
data = {
    "V1": -1.3598071336738,
    "V2": -0.0727811733098497,
    "V3": 2.53634673796914,
    "V4": 1.37815522427443,
    # ... other features
    "Amount": 149.62
}

response = requests.post(
    "http://localhost:8000/predict_anomaly",
    json=data,
    headers=headers
)
print(response.json())
```

### Dashboard Features

The Streamlit dashboard includes:

- **Anomaly Score Distribution**: Histogram of anomaly scores
- **SHAP Explanations**: Feature importance for individual predictions
- **Data Filtering**: Filter records by anomaly score, date, amount, etc.
- **Model Metrics**: Performance metrics and model information
- **Real-time Monitoring**: Live data drift detection

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python run_with_landing.py test

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Monitoring

The system includes comprehensive monitoring capabilities:

- **Data Drift Detection**: Compares reference vs. live data distributions
- **Model Performance Monitoring**: Tracks prediction accuracy over time
- **Alert System**: Configurable alerts for anomalies and drift
- **MLflow Integration**: All experiments and metrics are logged

## 🔧 Configuration

Key configuration options in `.env`:

- `MODEL_THRESHOLD`: Anomaly score threshold (default: 0.1)
- `DRIFT_THRESHOLD`: Data drift threshold (default: 0.1)
- `SECRET_KEY`: JWT secret key for authentication
- `MLFLOW_EXPERIMENT_NAME`: Name for MLflow experiments

## 🏗️ Architecture

The system follows clean architecture principles:

- **API Layer**: FastAPI handles HTTP requests and authentication
- **Business Logic**: Core anomaly detection logic in models/
- **Data Layer**: Data loading and preprocessing in data/
- **Infrastructure**: MLflow, monitoring, and utilities

## 🔒 Security

- JWT-based authentication with role-based access control
- Input validation using Pydantic models
- Secure configuration management with environment variables
- Rate limiting and CORS protection

## 📝 Development

### Code Quality

The project uses:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks for code quality

Setup pre-commit hooks:

```bash
pre-commit install
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

For questions, issues, or contributions, please:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Join our community discussions

---

**Built with ❤️ for production-ready anomaly detection** 