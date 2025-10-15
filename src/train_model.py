"""
TensorFlow model training script for predictive maintenance.

Implements deep learning model with proper versioning, checkpointing,
and integration with Azure ML Model Registry.
"""

import os
import argparse
import json
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictiveMaintenanceModel:
    """
    Deep learning model for equipment failure prediction.

    Uses a neural network architecture optimized for imbalanced
    time-series classification.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_units: list = [128, 64, 32],
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        """
        Initialize the model.

        Args:
            input_dim: Number of input features
            hidden_units: List of hidden layer sizes
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def build_model(self) -> keras.Model:
        """
        Build the neural network architecture.

        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,), name='input')

        # Hidden layers with batch normalization and dropout
        x = inputs
        for i, units in enumerate(self.hidden_units):
            x = layers.Dense(units, name=f'dense_{i}')(x)
            x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
            x = layers.Activation('relu', name=f'relu_{i}')(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i}')(x)

        # Output layer (binary classification)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='predictive_maintenance')

        # Compile with appropriate metrics for imbalanced data
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.AUC(name='pr_auc', curve='PR')
            ]
        )

        self.model = model
        logger.info(f"Built model with {model.count_params()} parameters")
        return model

    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = 50,
             batch_size: int = 128,
             class_weight: Dict = None) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            class_weight: Class weights for imbalanced data

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        # Calculate class weights if not provided
        if class_weight is None:
            neg_samples = np.sum(y_train == 0)
            pos_samples = np.sum(y_train == 1)
            total = len(y_train)

            class_weight = {
                0: (1 / neg_samples) * (total / 2.0),
                1: (1 / pos_samples) * (total / 2.0)
            }
            logger.info(f"Calculated class weights: {class_weight}")

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_pr_auc',
                patience=10,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath='models/checkpoints/best_model.keras',
                monitor='val_pr_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]

        # Create checkpoint directory
        os.makedirs('models/checkpoints', exist_ok=True)

        # Train model
        logger.info(f"Starting training for {epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callback_list,
            verbose=1
        )

        logger.info("Training completed")
        return self.history

    def evaluate(self,
                X_test: np.ndarray,
                y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate metrics
        metrics = {
            'accuracy': np.mean(y_pred == y_test),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        metrics['pr_auc'] = auc(recall, precision)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['precision'] = report['1']['precision']
        metrics['recall'] = report['1']['recall']
        metrics['f1_score'] = report['1']['f1-score']

        logger.info(f"Test Metrics: {metrics}")
        return metrics

    def plot_training_history(self, save_path: str = 'models/training_history.png'):
        """
        Plot training history.

        Args:
            save_path: Path to save plot
        """
        if self.history is None:
            logger.warning("No training history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # AUC
        axes[0, 1].plot(self.history.history['auc'], label='Train ROC-AUC')
        axes[0, 1].plot(self.history.history['val_auc'], label='Val ROC-AUC')
        axes[0, 1].set_title('ROC-AUC Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
        plt.close()

    def save_model(self, output_dir: str, version: str = None):
        """
        Save model with versioning.

        Args:
            output_dir: Directory to save model
            version: Model version (defaults to timestamp)
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_dir = os.path.join(output_dir, f'model_v{version}')
        os.makedirs(model_dir, exist_ok=True)

        # Save TensorFlow model
        model_path = os.path.join(model_dir, 'model.keras')
        self.model.save(model_path)

        # Save model metadata
        metadata = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'input_dim': self.input_dim,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'framework': 'tensorflow',
            'framework_version': tf.__version__
        }

        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model version {version} to {model_dir}")
        return model_dir


def load_data(train_path: str, test_path: str) -> Tuple:
    """
    Load and prepare data for training.

    Args:
        train_path: Path to training data
        test_path: Path to test data

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Separate features and target
    drop_cols = ['failure', 'timestamp', 'equipment_id', 'cycle']
    feature_cols = [col for col in train_df.columns if col not in drop_cols]

    X_train_full = train_df[feature_cols].values
    y_train_full = train_df['failure'].values

    X_test = test_df[feature_cols].values
    y_test = test_df['failure'].values

    # Split training into train/validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42
    )

    logger.info(f"Loaded data - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Train failure rate: {y_train.mean():.2%}")
    logger.info(f"Val failure rate: {y_val.mean():.2%}")
    logger.info(f"Test failure rate: {y_test.mean():.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main(args):
    """Main training pipeline."""

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        args.train_data,
        args.test_data
    )

    # Initialize model
    model = PredictiveMaintenanceModel(
        input_dim=X_train.shape[1],
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate
    )

    # Build and train
    model.build_model()
    model.model.summary()

    model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Evaluate
    metrics = model.evaluate(X_test, y_test)

    # Save results
    model.plot_training_history(save_path=args.output_dir + '/training_history.png')
    model_dir = model.save_model(args.output_dir, version=args.model_version)

    # Save metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Training complete! Model saved to {model_dir}")
    logger.info(f"Final metrics: {metrics}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train predictive maintenance model')

    parser.add_argument('--train-data', type=str,
                       default='data/processed/train_features.csv',
                       help='Path to training data')
    parser.add_argument('--test-data', type=str,
                       default='data/processed/test_features.csv',
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str,
                       default='models/',
                       help='Output directory for models')
    parser.add_argument('--model-version', type=str,
                       default=None,
                       help='Model version (default: timestamp)')
    parser.add_argument('--hidden-units', type=int, nargs='+',
                       default=[128, 64, 32],
                       help='Hidden layer sizes')
    parser.add_argument('--dropout-rate', type=float,
                       default=0.3,
                       help='Dropout rate')
    parser.add_argument('--learning-rate', type=float,
                       default=0.001,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int,
                       default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int,
                       default=128,
                       help='Batch size')

    args = parser.parse_args()
    main(args)
