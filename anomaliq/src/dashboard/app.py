"""
Streamlit dashboard for the Anomaliq system.
Interactive dashboard for anomaly detection visualization and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from typing import Dict, Any, Optional
import uuid

from src.data import CreditCardDataGenerator, generate_sample_record
from src.utils import get_settings, dashboard_logger

# Page configuration
st.set_page_config(
    page_title="Anomaliq Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize
settings = get_settings()
logger = dashboard_logger

def get_api_token() -> Optional[str]:
    """Get API token from session state or prompt user."""
    if 'api_token' not in st.session_state:
        st.session_state.api_token = None
    
    if not st.session_state.api_token:
        with st.sidebar:
            st.subheader("🔐 Authentication")
            username = st.text_input("Username", value="admin")
            password = st.text_input("Password", type="password", value="admin")
            
            if st.button("Login"):
                try:
                    response = requests.post(
                        f"http://localhost:{settings.api_port}/api/v1/auth/token",
                        data={"username": username, "password": password}
                    )
                    if response.status_code == 200:
                        st.session_state.api_token = response.json()["access_token"]
                        st.success("Logged in successfully!")
                        st.experimental_rerun()
                    else:
                        st.error("Login failed!")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
    
    return st.session_state.api_token

def predict_anomaly(record: Dict[str, Any], token: str) -> Optional[Dict[str, Any]]:
    """Make prediction request to API."""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        
        # Format request according to PredictionRequest model
        request_data = {
            "features": record,
            "metadata": {"record_id": str(uuid.uuid4())}
        }
        
        response = requests.post(
            f"http://localhost:{settings.api_port}/predict_anomaly",
            json=request_data,
            headers=headers
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample data for visualization."""
    generator = CreditCardDataGenerator()
    return generator.generate_dataset(n_samples=n_samples, fraud_ratio=0.01)

def plot_anomaly_distribution(df: pd.DataFrame):
    """Plot anomaly score distribution."""
    # Mock anomaly scores for demonstration
    np.random.seed(42)
    normal_scores = np.random.normal(-0.05, 0.1, len(df[df['Class'] == 0]))
    anomaly_scores = np.random.normal(-0.3, 0.15, len(df[df['Class'] == 1]))
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=normal_scores,
        name="Normal",
        opacity=0.7,
        nbinsx=50,
        marker_color="blue"
    ))
    
    fig.add_trace(go.Histogram(
        x=anomaly_scores,
        name="Anomaly",
        opacity=0.7,
        nbinsx=50,
        marker_color="red"
    ))
    
    fig.add_vline(x=-0.1, line_dash="dash", line_color="orange", 
                  annotation_text="Threshold")
    
    fig.update_layout(
        title="Anomaly Score Distribution",
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        barmode="overlay"
    )
    
    return fig

def plot_shap_explanation(shap_data: Dict[str, float]):
    """Plot SHAP feature importance."""
    features = list(shap_data.keys())
    values = list(shap_data.values())
    colors = ['red' if v < 0 else 'blue' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside"
    ))
    
    fig.update_layout(
        title="SHAP Feature Importance",
        xaxis_title="SHAP Value",
        yaxis_title="Features",
        height=400
    )
    
    return fig

def main():
    """Main dashboard application."""
    
    # Title
    st.title("🔍 Anomaliq Dashboard")
    st.markdown("**Production-ready anomaly detection system**")
    
    # Authentication
    token = get_api_token()
    if not token:
        st.warning("Please login to access the dashboard")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Real-time Prediction", "Data Analysis", "Model Monitoring"]
        )
        
        st.header("Settings")
        threshold = st.slider("Anomaly Threshold", -1.0, 1.0, -0.1, 0.01)
    
    # Main content based on selected page
    if page == "Real-time Prediction":
        st.header("🎯 Real-time Anomaly Prediction")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Transaction")
            
            # Generate sample data button
            if st.button("Generate Sample Transaction"):
                sample = generate_sample_record()
                st.session_state.sample_record = sample
            
            # Input fields
            if 'sample_record' in st.session_state:
                record = st.session_state.sample_record
                
                # Display key features
                amount = st.number_input("Amount", value=float(record.get('Amount', 100.0)))
                v14 = st.number_input("V14", value=float(record.get('V14', 0.0)))
                v4 = st.number_input("V4", value=float(record.get('V4', 0.0)))
                v12 = st.number_input("V12", value=float(record.get('V12', 0.0)))
                
                # Update record
                record['Amount'] = amount
                record['V14'] = v14
                record['V4'] = v4
                record['V12'] = v12
                
                # Predict button
                if st.button("🔍 Predict Anomaly", type="primary"):
                    with st.spinner("Making prediction..."):
                        result = predict_anomaly(record, token)
                        
                        if result:
                            st.session_state.prediction_result = result
        
        with col2:
            st.subheader("Prediction Result")
            
            if 'prediction_result' in st.session_state:
                result = st.session_state.prediction_result
                
                # Anomaly status
                is_anomaly = result.get('is_anomaly', False)
                anomaly_score = result.get('anomaly_score', 0.0)
                confidence = result.get('confidence', 0.0)
                
                if is_anomaly:
                    st.error(f"🚨 ANOMALY DETECTED")
                else:
                    st.success(f"✅ Normal Transaction")
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Anomaly Score", f"{anomaly_score:.3f}")
                with col_b:
                    st.metric("Confidence", f"{confidence:.3f}")
                with col_c:
                    st.metric("Threshold", f"{threshold:.3f}")
                
                # SHAP explanation
                if 'shap_explanation' in result and result['shap_explanation']:
                    st.subheader("🧠 Feature Importance (SHAP)")
                    shap_data = result['shap_explanation']['feature_importance']
                    fig_shap = plot_shap_explanation(shap_data)
                    st.plotly_chart(fig_shap, use_container_width=True)
    
    elif page == "Data Analysis":
        st.header("📊 Data Analysis")
        
        # Generate sample data for analysis
        with st.spinner("Generating sample data..."):
            df = generate_sample_data(1000)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Anomaly distribution
            fig_dist = plot_anomaly_distribution(df)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Summary statistics
            st.subheader("📈 Dataset Summary")
            total_transactions = len(df)
            fraud_count = (df['Class'] == 1).sum()
            fraud_rate = (fraud_count / total_transactions) * 100
            
            st.metric("Total Transactions", f"{total_transactions:,}")
            st.metric("Fraudulent", f"{fraud_count:,}")
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        
        # Feature correlation
        st.subheader("🔗 Feature Correlations")
        corr_features = ['V1', 'V2', 'V3', 'V4', 'V14', 'Amount']
        available_features = [f for f in corr_features if f in df.columns]
        
        if available_features:
            corr_matrix = df[available_features].corr()
            fig_corr = px.imshow(
                corr_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    elif page == "Model Monitoring":
        st.header("📈 Model Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            
            # Mock performance metrics
            st.metric("Accuracy", "95.2%", "↑ 1.2%")
            st.metric("Precision", "87.3%", "↓ 2.1%")
            st.metric("Recall", "91.8%", "↑ 0.5%")
            st.metric("F1 Score", "89.5%", "↓ 0.3%")
        
        with col2:
            st.subheader("System Health")
            
            st.metric("API Uptime", "99.9%", "0.0%")
            st.metric("Avg Response Time", "45ms", "↓ 5ms")
            st.metric("Predictions Today", "1,234", "↑ 156")
            st.metric("Anomalies Detected", "23", "↑ 3")
        
        # Performance over time (mock data)
        st.subheader("📊 Performance Trends")
        
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        performance_data = pd.DataFrame({
            'Date': dates,
            'Accuracy': np.random.normal(0.95, 0.02, len(dates)),
            'Precision': np.random.normal(0.87, 0.03, len(dates)),
            'Recall': np.random.normal(0.92, 0.025, len(dates))
        })
        
        fig_trends = px.line(
            performance_data,
            x='Date',
            y=['Accuracy', 'Precision', 'Recall'],
            title="Model Performance Over Time"
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Anomaliq** - Built with ❤️ for production-ready anomaly detection")

if __name__ == "__main__":
    main() 