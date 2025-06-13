"""
API package for the Anomaliq system.
FastAPI application with authentication and prediction endpoints.
"""

from .main import app
from .auth import get_current_user, require_admin
from .routes import router

__all__ = [
    "app",
    "get_current_user",
    "require_admin", 
    "router",
] 