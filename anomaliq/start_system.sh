#!/bin/bash

# Anomaliq System Startup Script
echo "🚀 Starting Anomaliq System..."

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

# Kill any existing processes
echo "🔄 Stopping existing services..."
pkill -f "uvicorn.*src.api.main:app" 2>/dev/null || true
pkill -f "streamlit.*src/dashboard/app.py" 2>/dev/null || true

sleep 2

# Start FastAPI in background
echo "🌐 Starting FastAPI server on port 8000..."
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for API to start
sleep 5

# Start Streamlit in background  
echo "📊 Starting Streamlit dashboard on port 8501..."
streamlit run src/dashboard/app.py --server.port 8501 &
DASHBOARD_PID=$!

# Wait for services to start
sleep 3

echo ""
echo "✅ Anomaliq System Started Successfully!"
echo ""
echo "🔗 Access Points:"
echo "   📱 Dashboard:     http://localhost:8501"
echo "   🔧 API Docs:      http://localhost:8000/docs"
echo "   💊 Health Check:  http://localhost:8000/health"
echo ""
echo "🔑 Login Credentials:"
echo "   👤 Admin: admin / admin"
echo "   👤 User:  user / user"
echo ""
echo "🛑 To stop the system, run: ./stop_system.sh"
echo ""

# Keep script running to show logs
wait 