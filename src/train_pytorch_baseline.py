"""
PyTorch baseline model for predictive maintenance.

Provides an alternative deep learning implementation for comparison with TensorFlow.
"""

import os
import argparse
import json
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, auc
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictiveMaintenanceNet(nn.Module):
    """
    PyTorch neural network for equipment failure prediction.

    Architecture matches the TensorFlow version for fair comparison.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_units: list = [128, 64, 32],
                 dropout_rate: float = 0.3):
        """
        Initialize the network.

        Args:
            input_dim: Number of input features
            hidden_units: List of hidden layer sizes
            dropout_rate: Dropout probability
        """
        super(PredictiveMaintenanceNet, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for i, units in enumerate(hidden_units):
            layers.extend([
                nn.Linear(prev_dim, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = units

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class PyTorchTrainer:
    """
    Trainer class for PyTorch model.

    Handles training loop, validation, and evaluation.
    """

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 0.001,
                 device: str = None):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            learning_rate: Learning rate
            device: Device to train on (cuda/cpu)
        """
        self.model = model
        self.learning_rate = learning_rate

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Initialize optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_auc': [], 'val_pr_auc': []
        }

        logger.info(f"Training on device: {self.device}")

    def train_epoch(self,
                   train_loader: DataLoader,
                   class_weights: torch.Tensor = None) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            class_weights: Weights for imbalanced classes

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(1)

            # Forward pass
            outputs = self.model(X_batch)

            # Calculate weighted loss if class weights provided
            if class_weights is not None:
                weights = torch.where(y_batch == 1,
                                    class_weights[1],
                                    class_weights[0])
                loss = (self.criterion(outputs, y_batch) * weights).mean()
            else:
                loss = self.criterion(outputs, y_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (loss, accuracy, roc_auc, pr_auc)
        """
        self.model.eval()
        total_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item()
                all_labels.extend(y_batch.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())

        # Calculate metrics
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        avg_loss = total_loss / len(val_loader)
        predicted_classes = (all_predictions > 0.5).astype(int)
        accuracy = np.mean(predicted_classes == all_labels)

        # ROC-AUC
        roc_auc = roc_auc_score(all_labels, all_predictions)

        # PR-AUC
        precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
        pr_auc = auc(recall, precision)

        return avg_loss, accuracy, roc_auc, pr_auc

    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             epochs: int = 50,
             patience: int = 10,
             class_weights: torch.Tensor = None) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            class_weights: Class weights for imbalanced data

        Returns:
            Training history
        """
        best_pr_auc = 0
        patience_counter = 0

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, class_weights)

            # Validate
            val_loss, val_acc, val_roc_auc, val_pr_auc = self.validate(val_loader)

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_roc_auc)
            self.history['val_pr_auc'].append(val_pr_auc)

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Val ROC-AUC: {val_roc_auc:.4f}, Val PR-AUC: {val_pr_auc:.4f}"
            )

            # Early stopping
            if val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc
                patience_counter = 0
                # Save best model
                os.makedirs('models/checkpoints', exist_ok=True)
                torch.save(self.model.state_dict(), 'models/checkpoints/best_model_pytorch.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Load best model
        self.model.load_state_dict(torch.load('models/checkpoints/best_model_pytorch.pth'))
        logger.info(f"Training completed. Best PR-AUC: {best_pr_auc:.4f}")

        return self.history

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                all_labels.extend(y_batch.numpy())
                all_predictions.extend(outputs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        # Calculate metrics
        y_pred = (all_predictions > 0.5).astype(int).flatten()
        metrics = {
            'accuracy': np.mean(y_pred == all_labels),
            'roc_auc': roc_auc_score(all_labels, all_predictions),
        }

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
        metrics['pr_auc'] = auc(recall, precision)

        # Classification report
        report = classification_report(all_labels, y_pred, output_dict=True)
        metrics['precision'] = report['1']['precision']
        metrics['recall'] = report['1']['recall']
        metrics['f1_score'] = report['1']['f1-score']

        logger.info(f"Test Metrics: {metrics}")
        return metrics


def prepare_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test,
                        batch_size: int = 128) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch data loaders.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        batch_size: Batch size

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def main(args):
    """Main training pipeline."""

    # Load data (reuse from TensorFlow script)
    from train_model import load_data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        args.train_data,
        args.test_data
    )

    # Create data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=args.batch_size
    )

    # Calculate class weights
    neg_samples = np.sum(y_train == 0)
    pos_samples = np.sum(y_train == 1)
    total = len(y_train)

    class_weights = torch.FloatTensor([
        (1 / neg_samples) * (total / 2.0),
        (1 / pos_samples) * (total / 2.0)
    ])

    # Initialize model
    model = PredictiveMaintenanceNet(
        input_dim=X_train.shape[1],
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout_rate
    )

    # Initialize trainer
    trainer = PyTorchTrainer(model, learning_rate=args.learning_rate)

    # Train
    history = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        patience=10,
        class_weights=class_weights
    )

    # Evaluate
    metrics = trainer.evaluate(test_loader)

    # Save model and metrics
    output_dir = os.path.join(args.output_dir, 'pytorch_model')
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))

    metadata = {
        'version': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'timestamp': datetime.now().isoformat(),
        'framework': 'pytorch',
        'framework_version': torch.__version__,
        'metrics': metrics
    }

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"PyTorch model saved to {output_dir}")
    logger.info(f"Final metrics: {metrics}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PyTorch baseline model')

    parser.add_argument('--train-data', type=str,
                       default='data/processed/train_features.csv',
                       help='Path to training data')
    parser.add_argument('--test-data', type=str,
                       default='data/processed/test_features.csv',
                       help='Path to test data')
    parser.add_argument('--output-dir', type=str,
                       default='models/',
                       help='Output directory for models')
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
