#!/usr/bin/env python3
"""
Unified entry point for the Anomaliq anomaly detection system.
This script provides a single interface to launch different components.
"""

import click
import subprocess
import sys
import os
from typing import Optional


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """Anomaliq - Production-ready anomaly detection system."""
    pass


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind the API server')
@click.option('--port', default=8000, help='Port to bind the API server')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def api(host: str, port: int, reload: bool):
    """Launch the FastAPI backend server."""
    click.echo(f"🚀 Starting Anomaliq API server on {host}:{port}")
    
    cmd = [
        "uvicorn",
        "src.api.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        click.echo("\n👋 API server stopped")
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Error starting API server: {e}")
        sys.exit(1)


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind the dashboard')
@click.option('--port', default=8501, help='Port to bind the dashboard')
def dashboard(host: str, port: int):
    """Launch the Streamlit dashboard."""
    click.echo(f"📊 Starting Anomaliq dashboard on {host}:{port}")
    
    cmd = [
        "streamlit", "run",
        "src/dashboard/app.py",
        "--server.address", host,
        "--server.port", str(port)
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        click.echo("\n👋 Dashboard stopped")
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Error starting dashboard: {e}")
        sys.exit(1)


@cli.command()
@click.option('--data-path', help='Path to training data CSV file')
@click.option('--experiment-name', help='MLflow experiment name')
def train(data_path: Optional[str], experiment_name: Optional[str]):
    """Train the anomaly detection model."""
    click.echo("🤖 Training anomaly detection model...")
    
    cmd = ["python", "-m", "src.models.train_model"]
    
    if data_path:
        cmd.extend(["--data-path", data_path])
    
    if experiment_name:
        cmd.extend(["--experiment-name", experiment_name])
    
    try:
        subprocess.run(cmd, check=True)
        click.echo("✅ Model training completed successfully")
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Error training model: {e}")
        sys.exit(1)


@cli.command()
def mlflow():
    """Launch MLflow UI for experiment tracking."""
    click.echo("📈 Starting MLflow UI...")
    
    cmd = ["mlflow", "ui", "--backend-store-uri", "./mlruns"]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        click.echo("\n👋 MLflow UI stopped")
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Error starting MLflow UI: {e}")
        sys.exit(1)


@cli.command()
def test():
    """Run the test suite."""
    click.echo("🧪 Running test suite...")
    
    cmd = ["python", "-m", "pytest", "tests/", "-v"]
    
    try:
        subprocess.run(cmd, check=True)
        click.echo("✅ All tests passed")
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Some tests failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--reference-path', help='Path to reference data CSV file')
@click.option('--live-path', help='Path to live data CSV file')
def monitor(reference_path: Optional[str], live_path: Optional[str]):
    """Run data drift monitoring."""
    click.echo("📊 Running data drift monitoring...")
    
    cmd = ["python", "-m", "src.monitoring.drift_detection"]
    
    if reference_path:
        cmd.extend(["--reference-path", reference_path])
    
    if live_path:
        cmd.extend(["--live-path", live_path])
    
    try:
        subprocess.run(cmd, check=True)
        click.echo("✅ Monitoring completed")
    except subprocess.CalledProcessError as e:
        click.echo(f"❌ Error in monitoring: {e}")
        sys.exit(1)


@cli.command()
def full():
    """Launch both API and dashboard simultaneously."""
    click.echo("🚀 Starting full Anomaliq system...")
    
    # Start API in background
    api_process = subprocess.Popen([
        "uvicorn", "src.api.main:app", 
        "--host", "0.0.0.0", "--port", "8000"
    ])
    
    # Start dashboard in background
    dashboard_process = subprocess.Popen([
        "streamlit", "run", "src/dashboard/app.py",
        "--server.address", "0.0.0.0", "--server.port", "8501"
    ])
    
    click.echo("🌐 API running on http://localhost:8000")
    click.echo("📊 Dashboard running on http://localhost:8501")
    click.echo("Press Ctrl+C to stop both services")
    
    try:
        # Wait for both processes
        api_process.wait()
        dashboard_process.wait()
    except KeyboardInterrupt:
        click.echo("\n🛑 Stopping services...")
        api_process.terminate()
        dashboard_process.terminate()
        api_process.wait()
        dashboard_process.wait()
        click.echo("👋 All services stopped")


if __name__ == '__main__':
    cli() 