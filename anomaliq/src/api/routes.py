"""
Additional API routes for the Anomaliq system.
Authentication and admin functionality.
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Dict, Any

from src.api.auth import authenticate_user, create_access_token, Token, User, require_admin
from src.utils import get_settings, api_logger

router = APIRouter()
settings = get_settings()

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return access token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        api_logger.warning(f"Failed login attempt for username: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}, expires_delta=access_token_expires
    )
    
    api_logger.info(f"User {user.username} logged in successfully")
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/auth/me", response_model=User)
async def read_users_me(current_user: User = Depends(require_admin)):
    """Get current user information."""
    return current_user

@router.get("/admin/metrics")
async def get_admin_metrics(current_user: User = Depends(require_admin)):
    """Get system metrics (admin only)."""
    # Mock metrics - in production, get real system metrics
    return {
        "total_predictions": 1234,
        "anomalies_detected": 56,
        "api_uptime": "99.9%",
        "model_accuracy": 0.95,
        "active_users": 42
    } 