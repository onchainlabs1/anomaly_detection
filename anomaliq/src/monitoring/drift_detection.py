#!/usr/bin/env python3
"""
Data drift detection for the Anomaliq system.
Monitors data distribution changes between reference and live data.
"""

import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
from evidently.report import Report
from evidently.metrics import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset

from src.data import DataLoader
from src.utils import get_settings, get_monitoring_config, monitoring_logger, log_data_drift


class DataDriftDetector:
    """Detects data drift between reference and live datasets."""
    
    def __init__(self):
        self.settings = get_settings()
        self.monitoring_config = get_monitoring_config()
        self.logger = monitoring_logger
    
    def kolmogorov_smirnov_test(
        self, 
        reference_data: pd.Series, 
        live_data: pd.Series
    ) -> Tuple[float, float]:
        """Perform Kolmogorov-Smirnov test for distribution comparison."""
        try:
            statistic, p_value = stats.ks_2samp(reference_data, live_data)
            return float(statistic), float(p_value)
        except Exception as e:
            self.logger.warning(f"KS test failed: {str(e)}")
            return 0.0, 1.0
    
    def wasserstein_distance(
        self, 
        reference_data: pd.Series, 
        live_data: pd.Series
    ) -> float:
        """Calculate Wasserstein distance between distributions."""
        try:
            distance = stats.wasserstein_distance(reference_data, live_data)
            return float(distance)
        except Exception as e:
            self.logger.warning(f"Wasserstein distance calculation failed: {str(e)}")
            return 0.0
    
    def population_stability_index(
        self, 
        reference_data: pd.Series, 
        live_data: pd.Series,
        bins: int = 10
    ) -> float:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference_data, bins=bins)
            
            # Calculate frequencies for both datasets
            ref_freq, _ = np.histogram(reference_data, bins=bin_edges)
            live_freq, _ = np.histogram(live_data, bins=bin_edges)
            
            # Convert to proportions and avoid division by zero
            ref_prop = ref_freq / len(reference_data)
            live_prop = live_freq / len(live_data)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_prop = np.maximum(ref_prop, epsilon)
            live_prop = np.maximum(live_prop, epsilon)
            
            # Calculate PSI
            psi = np.sum((live_prop - ref_prop) * np.log(live_prop / ref_prop))
            return float(psi)
            
        except Exception as e:
            self.logger.warning(f"PSI calculation failed: {str(e)}")
            return 0.0
    
    def detect_feature_drift(
        self, 
        reference_data: pd.DataFrame, 
        live_data: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """Detect drift for individual features."""
        drift_results = {}
        
        # Get common numeric columns
        numeric_columns = reference_data.select_dtypes(include=[np.number]).columns
        common_columns = set(numeric_columns) & set(live_data.columns)
        
        for column in common_columns:
            ref_col = reference_data[column].dropna()
            live_col = live_data[column].dropna()
            
            if len(ref_col) == 0 or len(live_col) == 0:
                continue
            
            # Calculate multiple drift metrics
            ks_stat, ks_p_value = self.kolmogorov_smirnov_test(ref_col, live_col)
            wasserstein_dist = self.wasserstein_distance(ref_col, live_col)
            psi = self.population_stability_index(ref_col, live_col)
            
            # Determine if drift is significant
            is_drift = (
                ks_stat > self.settings.drift_threshold or 
                psi > self.monitoring_config.ALERT_THRESHOLD
            )
            
            drift_results[column] = {
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'wasserstein_distance': wasserstein_dist,
                'psi': psi,
                'is_drift': is_drift,
                'drift_severity': self._calculate_drift_severity(ks_stat, psi)
            }
            
            # Log drift if detected
            if is_drift:
                log_data_drift(column, max(ks_stat, psi), self.settings.drift_threshold)
        
        return drift_results
    
    def _calculate_drift_severity(self, ks_stat: float, psi: float) -> str:
        """Calculate drift severity level."""
        max_metric = max(ks_stat, psi)
        
        if max_metric < 0.1:
            return "low"
        elif max_metric < 0.2:
            return "medium"
        else:
            return "high"
    
    def generate_drift_report(
        self, 
        reference_data: pd.DataFrame, 
        live_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate comprehensive drift report using Evidently."""
        try:
            # Create test suite for drift detection
            suite = TestSuite(tests=[
                DataDriftTestPreset(),
            ])
            
            suite.run(
                reference_data=reference_data,
                current_data=live_data
            )
            
            # Create report with drift metrics
            report = Report(metrics=[
                DataDriftPreset(),
            ])
            
            report.run(
                reference_data=reference_data,
                current_data=live_data
            )
            
            # Extract results
            results = {
                'test_suite': suite.json(),
                'metrics': report.json()
            }
            
            return {
                'evidently_report': results,
                'timestamp': pd.Timestamp.now().isoformat(),
                'reference_size': len(reference_data),
                'live_size': len(live_data)
            }
            
        except Exception as e:
            self.logger.error(f"Evidently report generation failed: {str(e)}")
            return {
                'error': str(e),
                'timestamp': pd.Timestamp.now().isoformat()
            }
    
    def run_drift_detection(
        self, 
        reference_path: Optional[str] = None,
        live_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run complete drift detection pipeline."""
        self.logger.info("Starting data drift detection")
        
        # Load data
        data_loader = DataLoader()
        
        if reference_path:
            reference_data = data_loader.load_csv(reference_path)
        else:
            reference_data = data_loader.load_reference_data()
        
        if live_path:
            live_data = data_loader.load_csv(live_path)
        else:
            live_data = data_loader.load_live_data()
        
        if reference_data.empty or live_data.empty:
            self.logger.error("Reference or live data is empty")
            return {"error": "Missing data for drift detection"}
        
        self.logger.info(f"Reference data: {len(reference_data)} samples")
        self.logger.info(f"Live data: {len(live_data)} samples")
        
        # Detect feature-level drift
        feature_drift = self.detect_feature_drift(reference_data, live_data)
        
        # Generate comprehensive report
        drift_report = self.generate_drift_report(reference_data, live_data)
        
        # Calculate overall drift summary
        drift_features = [col for col, results in feature_drift.items() if results['is_drift']]
        overall_drift = len(drift_features) > 0
        
        summary = {
            'overall_drift_detected': overall_drift,
            'n_features_with_drift': len(drift_features),
            'drift_features': drift_features,
            'total_features_analyzed': len(feature_drift),
            'drift_percentage': (len(drift_features) / len(feature_drift)) * 100 if feature_drift else 0
        }
        
        # Compile final results
        results = {
            'summary': summary,
            'feature_drift': feature_drift,
            'evidently_report': drift_report,
            'timestamp': pd.Timestamp.now().isoformat(),
            'settings': {
                'drift_threshold': self.settings.drift_threshold,
                'alert_threshold': self.monitoring_config.ALERT_THRESHOLD
            }
        }
        
        # Log summary
        if overall_drift:
            self.logger.warning(f"Data drift detected in {len(drift_features)} features: {drift_features}")
        else:
            self.logger.info("No significant data drift detected")
        
        return results


def main():
    """Main function for command line drift detection."""
    parser = argparse.ArgumentParser(description="Detect data drift")
    parser.add_argument("--reference-path", type=str, help="Path to reference data CSV")
    parser.add_argument("--live-path", type=str, help="Path to live data CSV")
    parser.add_argument("--output", type=str, help="Output file for drift report")
    
    args = parser.parse_args()
    
    # Initialize drift detector
    detector = DataDriftDetector()
    
    # Run drift detection
    try:
        results = detector.run_drift_detection(
            reference_path=args.reference_path,
            live_path=args.live_path
        )
        
        # Save results if output specified
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Drift report saved to {args.output}")
        
        # Print summary
        summary = results.get('summary', {})
        print(f"\n📊 Data Drift Detection Summary")
        print(f"Overall drift detected: {'⚠️  YES' if summary.get('overall_drift_detected') else '✅ NO'}")
        print(f"Features with drift: {summary.get('n_features_with_drift', 0)}/{summary.get('total_features_analyzed', 0)}")
        print(f"Drift percentage: {summary.get('drift_percentage', 0):.1f}%")
        
        if summary.get('drift_features'):
            print(f"Affected features: {', '.join(summary['drift_features'])}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Drift detection failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main()) 