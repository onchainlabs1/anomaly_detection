#!/bin/bash

# Anomaliq System Stop Script
echo "🛑 Stopping Anomaliq System..."

# Stop FastAPI
echo "🔄 Stopping FastAPI server..."
pkill -f "uvicorn.*src.api.main:app" 2>/dev/null || true

# Stop Streamlit
echo "🔄 Stopping Streamlit dashboard..."
pkill -f "streamlit.*src/dashboard/app.py" 2>/dev/null || true

sleep 2

echo "✅ Anomaliq System Stopped Successfully!"
echo "" 