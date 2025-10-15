"""
Model evaluation script for predictive maintenance models.

Provides comprehensive evaluation metrics, visualizations, and
performance reports for trained models.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, roc_auc_score
)
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load trained TensorFlow model."""
    model = tf.keras.models.load_model(model_path)
    logger.info(f"Loaded model from {model_path}")
    return model


def load_test_data(data_path: str):
    """Load and prepare test data."""
    df = pd.read_csv(data_path)

    drop_cols = ['failure', 'timestamp', 'equipment_id', 'cycle']
    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols].values
    y = df['failure'].values

    logger.info(f"Loaded {len(X)} test samples")
    return X, y, feature_cols


def evaluate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive evaluation metrics."""
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = np.mean(y_pred == y_true)

    # ROC-AUC
    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    metrics['pr_auc'] = auc(recall, precision)

    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['precision'] = report['1']['precision']
    metrics['recall'] = report['1']['recall']
    metrics['f1_score'] = report['1']['f1-score']
    metrics['support_positive'] = int(report['1']['support'])
    metrics['support_negative'] = int(report['0']['support'])

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['true_positives'] = int(cm[1, 1])

    return metrics, report, cm


def plot_confusion_matrix(cm, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Failure'],
                yticklabels=['Normal', 'Failure'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_roc_curve(y_true, y_pred_proba, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curve to {save_path}")


def plot_precision_recall_curve(y_true, y_pred_proba, save_path):
    """Plot and save Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved PR curve to {save_path}")


def plot_prediction_distribution(y_pred_proba, y_true, save_path):
    """Plot distribution of prediction probabilities."""
    plt.figure(figsize=(10, 6))

    # Separate by true class
    failures = y_pred_proba[y_true == 1]
    normal = y_pred_proba[y_true == 0]

    plt.hist(normal, bins=50, alpha=0.7, label='Normal', color='green')
    plt.hist(failures, bins=50, alpha=0.7, label='Failure', color='red')
    plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved prediction distribution to {save_path}")


def generate_evaluation_report(metrics, report, output_path):
    """Generate comprehensive evaluation report."""
    report_lines = [
        "=" * 70,
        "PREDICTIVE MAINTENANCE MODEL - EVALUATION REPORT",
        "=" * 70,
        "",
        "OVERALL METRICS",
        "-" * 70,
        f"Accuracy:        {metrics['accuracy']:.4f}",
        f"ROC-AUC:         {metrics['roc_auc']:.4f}",
        f"PR-AUC:          {metrics['pr_auc']:.4f}",
        f"Precision:       {metrics['precision']:.4f}",
        f"Recall:          {metrics['recall']:.4f}",
        f"F1-Score:        {metrics['f1_score']:.4f}",
        "",
        "CONFUSION MATRIX",
        "-" * 70,
        f"True Negatives:  {metrics['true_negatives']}",
        f"False Positives: {metrics['false_positives']}",
        f"False Negatives: {metrics['false_negatives']}",
        f"True Positives:  {metrics['true_positives']}",
        "",
        "CLASS DISTRIBUTION",
        "-" * 70,
        f"Normal samples:  {metrics['support_negative']}",
        f"Failure samples: {metrics['support_positive']}",
        "",
        "=" * 70,
    ]

    report_text = "\n".join(report_lines)

    with open(output_path, 'w') as f:
        f.write(report_text)

    logger.info(f"Saved evaluation report to {output_path}")
    print("\n" + report_text)


def main(args):
    """Main evaluation pipeline."""

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and data
    model = load_model(args.model_path)
    X_test, y_test, feature_names = load_test_data(args.test_data)

    # Get predictions
    logger.info("Generating predictions...")
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > args.threshold).astype(int)

    # Calculate metrics
    metrics, report, cm = evaluate_metrics(y_test, y_pred, y_pred_proba)

    # Generate visualizations
    plot_confusion_matrix(cm, os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_roc_curve(y_test, y_pred_proba, os.path.join(args.output_dir, 'roc_curve.png'))
    plot_precision_recall_curve(y_test, y_pred_proba, os.path.join(args.output_dir, 'pr_curve.png'))
    plot_prediction_distribution(y_pred_proba, y_test, os.path.join(args.output_dir, 'prediction_distribution.png'))

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Generate report
    report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
    generate_evaluation_report(metrics, report, report_path)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate predictive maintenance model')

    parser.add_argument('--model-path', type=str,
                       required=True,
                       help='Path to trained model (.keras file)')
    parser.add_argument('--test-data', type=str,
                       default='data/processed/test_features.csv',
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str,
                       default='models/evaluation/',
                       help='Output directory for evaluation results')
    parser.add_argument('--threshold', type=float,
                       default=0.5,
                       help='Classification threshold')

    args = parser.parse_args()
    main(args)
