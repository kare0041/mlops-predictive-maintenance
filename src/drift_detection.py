"""
Data drift detection module for predictive maintenance.

Monitors input data distribution shifts using statistical tests
(Kolmogorov-Smirnov test) and triggers retraining when drift is detected.
"""

import os
import json
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects distribution drift in sensor data using statistical tests.

    Uses Kolmogorov-Smirnov test to compare reference and current distributions.
    """

    def __init__(self,
                 reference_data_path: str = None,
                 significance_level: float = 0.05,
                 drift_threshold: float = 0.3):
        """
        Initialize drift detector.

        Args:
            reference_data_path: Path to reference (training) data
            significance_level: P-value threshold for KS test
            drift_threshold: Proportion of features drifted to trigger alert
        """
        self.reference_data = None
        self.significance_level = significance_level
        self.drift_threshold = drift_threshold
        self.feature_stats = {}

        if reference_data_path:
            self.load_reference_data(reference_data_path)

    def load_reference_data(self, data_path: str):
        """
        Load reference data for comparison.

        Args:
            data_path: Path to reference CSV file
        """
        self.reference_data = pd.read_csv(data_path)

        # Exclude non-feature columns
        exclude_cols = ['failure', 'timestamp', 'equipment_id', 'cycle']
        self.feature_cols = [
            col for col in self.reference_data.columns
            if col not in exclude_cols
        ]

        # Calculate reference statistics
        for col in self.feature_cols:
            self.feature_stats[col] = {
                'mean': self.reference_data[col].mean(),
                'std': self.reference_data[col].std(),
                'min': self.reference_data[col].min(),
                'max': self.reference_data[col].max(),
                'median': self.reference_data[col].median()
            }

        logger.info(f"Loaded reference data: {len(self.reference_data)} samples, "
                   f"{len(self.feature_cols)} features")

    def detect_drift_ks_test(self,
                            current_data: pd.DataFrame) -> Dict:
        """
        Perform Kolmogorov-Smirnov test for each feature.

        Args:
            current_data: Current production data

        Returns:
            Dictionary with drift test results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded")

        drift_results = {}
        drifted_features = []

        for col in self.feature_cols:
            if col not in current_data.columns:
                logger.warning(f"Feature {col} not found in current data")
                continue

            # Perform KS test
            statistic, p_value = stats.ks_2samp(
                self.reference_data[col].dropna(),
                current_data[col].dropna()
            )

            is_drifted = p_value < self.significance_level

            drift_results[col] = {
                'ks_statistic': float(statistic),
                'p_value': float(p_value),
                'is_drifted': bool(is_drifted),
                'reference_mean': float(self.feature_stats[col]['mean']),
                'current_mean': float(current_data[col].mean()),
                'mean_shift': float(
                    abs(current_data[col].mean() - self.feature_stats[col]['mean'])
                )
            }

            if is_drifted:
                drifted_features.append(col)

        # Summary
        total_features = len(self.feature_cols)
        num_drifted = len(drifted_features)
        drift_proportion = num_drifted / total_features

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_features': total_features,
            'drifted_features_count': num_drifted,
            'drift_proportion': drift_proportion,
            'drifted_features': drifted_features,
            'drift_detected': drift_proportion >= self.drift_threshold,
            'significance_level': self.significance_level
        }

        return {'summary': summary, 'feature_results': drift_results}

    def detect_drift_statistical(self,
                                current_data: pd.DataFrame,
                                method: str = 'mean') -> Dict:
        """
        Detect drift using simple statistical comparisons.

        Args:
            current_data: Current production data
            method: 'mean' or 'distribution'

        Returns:
            Dictionary with drift results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not loaded")

        drift_results = {}
        drifted_features = []

        for col in self.feature_cols:
            if col not in current_data.columns:
                continue

            ref_mean = self.feature_stats[col]['mean']
            ref_std = self.feature_stats[col]['std']
            curr_mean = current_data[col].mean()

            # Check if current mean is outside 3 sigma
            z_score = abs((curr_mean - ref_mean) / (ref_std + 1e-10))
            is_drifted = z_score > 3

            drift_results[col] = {
                'z_score': float(z_score),
                'is_drifted': bool(is_drifted),
                'reference_mean': float(ref_mean),
                'current_mean': float(curr_mean)
            }

            if is_drifted:
                drifted_features.append(col)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'drifted_features': drifted_features,
            'drift_detected': len(drifted_features) > 0
        }

        return {'summary': summary, 'feature_results': drift_results}

    def visualize_drift(self,
                       current_data: pd.DataFrame,
                       output_dir: str = 'models/drift_reports/',
                       top_n: int = 6):
        """
        Create visualization comparing reference and current distributions.

        Args:
            current_data: Current production data
            output_dir: Directory to save plots
            top_n: Number of top drifted features to plot
        """
        os.makedirs(output_dir, exist_ok=True)

        # Get drift results
        results = self.detect_drift_ks_test(current_data)

        # Sort features by KS statistic
        sorted_features = sorted(
            results['feature_results'].items(),
            key=lambda x: x[1]['ks_statistic'],
            reverse=True
        )[:top_n]

        # Create subplots
        n_rows = (top_n + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, n_rows * 4))
        axes = axes.flatten()

        for idx, (feature, result) in enumerate(sorted_features):
            ax = axes[idx]

            # Plot distributions
            ref_data = self.reference_data[feature].dropna()
            curr_data = current_data[feature].dropna()

            ax.hist(ref_data, bins=50, alpha=0.5, label='Reference', color='blue', density=True)
            ax.hist(curr_data, bins=50, alpha=0.5, label='Current', color='red', density=True)

            # Add vertical lines for means
            ax.axvline(ref_data.mean(), color='blue', linestyle='--', linewidth=2)
            ax.axvline(curr_data.mean(), color='red', linestyle='--', linewidth=2)

            drift_status = "DRIFT DETECTED" if result['is_drifted'] else "No Drift"
            ax.set_title(f"{feature}\n{drift_status} (KS={result['ks_statistic']:.3f}, p={result['p_value']:.4f})")
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'drift_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved drift visualization to {plot_path}")

    def generate_drift_report(self,
                             current_data: pd.DataFrame,
                             output_dir: str = 'models/drift_reports/') -> str:
        """
        Generate comprehensive drift detection report.

        Args:
            current_data: Current production data
            output_dir: Directory to save report

        Returns:
            Path to saved report
        """
        os.makedirs(output_dir, exist_ok=True)

        # Perform drift detection
        results = self.detect_drift_ks_test(current_data)

        # Save JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f'drift_report_{timestamp}.json')

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Create text report
        summary = results['summary']
        txt_path = os.path.join(output_dir, f'drift_report_{timestamp}.txt')

        report_lines = [
            "=" * 70,
            "DATA DRIFT DETECTION REPORT",
            "=" * 70,
            f"Timestamp: {summary['timestamp']}",
            f"Significance Level: {summary['significance_level']}",
            "",
            "SUMMARY",
            "-" * 70,
            f"Total Features Monitored: {summary['total_features']}",
            f"Features with Drift: {summary['drifted_features_count']}",
            f"Drift Proportion: {summary['drift_proportion']:.2%}",
            f"Drift Detected: {'YES - RETRAINING RECOMMENDED' if summary['drift_detected'] else 'NO'}",
            "",
            "DRIFTED FEATURES",
            "-" * 70
        ]

        if summary['drifted_features']:
            for feature in summary['drifted_features']:
                result = results['feature_results'][feature]
                report_lines.append(
                    f"  - {feature}: KS={result['ks_statistic']:.4f}, "
                    f"p-value={result['p_value']:.6f}, "
                    f"Mean Shift={result['mean_shift']:.4f}"
                )
        else:
            report_lines.append("  (None)")

        report_lines.append("\n" + "=" * 70)

        report_text = "\n".join(report_lines)

        with open(txt_path, 'w') as f:
            f.write(report_text)

        logger.info(f"Saved drift report to {json_path}")
        print("\n" + report_text)

        # Generate visualization
        self.visualize_drift(current_data, output_dir)

        return json_path


def monitor_drift(reference_data_path: str,
                 current_data_path: str,
                 output_dir: str = 'models/drift_reports/') -> bool:
    """
    Convenience function to monitor drift.

    Args:
        reference_data_path: Path to reference data
        current_data_path: Path to current data
        output_dir: Output directory for reports

    Returns:
        True if drift detected, False otherwise
    """
    detector = DriftDetector(reference_data_path=reference_data_path)
    current_data = pd.read_csv(current_data_path)

    detector.generate_drift_report(current_data, output_dir)

    results = detector.detect_drift_ks_test(current_data)
    return results['summary']['drift_detected']


if __name__ == "__main__":
    # Example usage
    detector = DriftDetector(
        reference_data_path='data/processed/train_features.csv',
        significance_level=0.05,
        drift_threshold=0.2
    )

    # Simulate current data (using test set as example)
    current_data = pd.read_csv('data/processed/test_features.csv')

    # Generate drift report
    detector.generate_drift_report(current_data)
