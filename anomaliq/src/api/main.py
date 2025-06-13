"""
FastAPI main application module for Anomaliq.
Handles API routes and model serving.
"""

from typing import Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, status, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from datetime import datetime

from src.models import AnomalyPredictor
from src.utils import (
    get_settings,
    api_logger,
    log_api_request,
    log_anomaly_prediction
)
from src.api.auth import (
    Token,
    UserInDB,
    get_current_active_user,
    authenticate_user,
    create_access_token
)

# Initialize settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="Anomaliq API",
    description="API for anomaly detection in credit card transactions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor: Optional[AnomalyPredictor] = None

class PredictionRequest(BaseModel):
    """Model for prediction request data."""
    features: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    """Model for prediction response."""
    prediction: bool
    anomaly_score: float
    timestamp: str
    record_id: str
    metadata: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on application startup."""
    global predictor
    
    api_logger.info("Starting Anomaliq API")
    
    try:
        # Initialize predictor
        api_logger.info("Initializing anomaly predictor")
        predictor = AnomalyPredictor()
        
        # Load model from MLflow
        predictor.load_model_by_name(settings.mlflow_model_name)
        
        api_logger.info("Anomaliq API started successfully")
    
    except Exception as e:
        api_logger.error(f"Failed to initialize API: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize API"
        )

@app.post("/api/v1/auth/token", response_model=Token)
async def login_for_access_token(username: str = Form(...), password: str = Form(...)):
    """Login endpoint to get access token."""
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict_anomaly", response_model=PredictionResponse)
async def predict_anomaly(
    request: PredictionRequest,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Predict if a transaction is anomalous."""
    try:
        # Log request
        api_logger.info(f"Anomaly prediction request from user: {current_user.username}")
        
        # Validate predictor
        if not predictor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Predictor not initialized"
            )
        
        # Make prediction
        start_time = datetime.now()
        result = predictor.predict_single(request.features)
        
        # Create response
        response = PredictionResponse(
            prediction=result['is_anomaly'],
            anomaly_score=result['anomaly_score'],
            timestamp=datetime.now().isoformat(),
            record_id=request.metadata.get("record_id", "unknown") if request.metadata else "unknown",
            metadata=request.metadata
        )
        
        # Log prediction
        log_anomaly_prediction(
            record_id=response.record_id,
            anomaly_score=response.anomaly_score,
            is_anomaly=response.prediction
        )
        
        # Log API metrics
        end_time = datetime.now()
        log_api_request(
            method="POST",
            endpoint="/predict_anomaly",
            status_code=200,
            response_time=(end_time - start_time).total_seconds() * 1000
        )
        
        return response
    
    except Exception as e:
        api_logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()} 