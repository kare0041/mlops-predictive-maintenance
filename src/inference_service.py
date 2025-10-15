"""
Production inference service for predictive maintenance.

Flask REST API with Prometheus monitoring for model serving.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['model_version', 'prediction']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction request',
    ['model_version']
)

PREDICTION_CONFIDENCE = Histogram(
    'prediction_confidence',
    'Confidence score of predictions',
    ['prediction']
)

MODEL_LOAD_TIME = Gauge(
    'model_load_time_seconds',
    'Time taken to load the model'
)

HEALTH_STATUS = Gauge(
    'service_health_status',
    'Health status of the service (1=healthy, 0=unhealthy)'
)


class PredictiveMaintenanceService:
    """
    Production inference service for equipment failure prediction.

    Handles model loading, preprocessing, and batch predictions.
    """

    def __init__(self, model_dir: str, artifacts_dir: str = 'models/artifacts/'):
        """
        Initialize the service.

        Args:
            model_dir: Directory containing the trained model
            artifacts_dir: Directory with preprocessing artifacts
        """
        self.model_dir = model_dir
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metadata = None

        self._load_model()
        self._load_artifacts()

        HEALTH_STATUS.set(1)

    def _load_model(self):
        """Load the TensorFlow model."""
        start_time = time.time()

        try:
            model_path = os.path.join(self.model_dir, 'model.keras')
            self.model = tf.keras.models.load_model(model_path)

            # Load metadata
            metadata_path = os.path.join(self.model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)

            load_time = time.time() - start_time
            MODEL_LOAD_TIME.set(load_time)

            logger.info(f"Model loaded in {load_time:.2f}s from {model_path}")
            logger.info(f"Model version: {self.metadata.get('version', 'unknown')}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            HEALTH_STATUS.set(0)
            raise

    def _load_artifacts(self):
        """Load preprocessing artifacts (scaler, feature names)."""
        try:
            # Load scaler
            scaler_path = os.path.join(self.artifacts_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")

            # Load feature names
            features_path = os.path.join(self.artifacts_dir, 'feature_names.pkl')
            if os.path.exists(features_path):
                self.feature_names = joblib.load(features_path)
                logger.info(f"Loaded {len(self.feature_names)} feature names")

        except Exception as e:
            logger.warning(f"Could not load artifacts: {e}")

    def preprocess_input(self, data: dict) -> np.ndarray:
        """
        Preprocess input data for prediction.

        Args:
            data: Dictionary of sensor readings

        Returns:
            Preprocessed feature array
        """
        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Feature engineering (simplified - in production, use full pipeline)
        # This is a basic example - you would need to replicate the full
        # feature engineering pipeline from feature_engineering.py

        # Extract base features
        base_features = [
            'temperature', 'vibration', 'pressure',
            'rotation_speed', 'current', 'operating_hours'
        ]

        # Check for missing required features
        missing = set(base_features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # For this example, we'll use basic features
        # In production, you'd apply full feature engineering
        X = df[base_features].values

        return X

    def predict(self, data: dict) -> dict:
        """
        Make a prediction for a single equipment reading.

        Args:
            data: Dictionary containing sensor readings

        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()

        try:
            # Preprocess input
            X = self.preprocess_input(data)

            # Make prediction
            prediction_proba = self.model.predict(X, verbose=0)[0][0]
            prediction_class = int(prediction_proba > 0.5)

            # Determine risk level
            if prediction_proba < 0.3:
                risk_level = "LOW"
            elif prediction_proba < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            # Prepare response
            result = {
                'equipment_id': data.get('equipment_id', 'unknown'),
                'prediction': prediction_class,
                'probability': float(prediction_proba),
                'risk_level': risk_level,
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_version': self.metadata.get('version', 'unknown'),
                'inference_time_ms': (time.time() - start_time) * 1000
            }

            # Update Prometheus metrics
            model_version = self.metadata.get('version', 'unknown')
            PREDICTION_COUNTER.labels(
                model_version=model_version,
                prediction=str(prediction_class)
            ).inc()

            PREDICTION_LATENCY.labels(
                model_version=model_version
            ).observe(time.time() - start_time)

            PREDICTION_CONFIDENCE.labels(
                prediction=str(prediction_class)
            ).observe(prediction_proba)

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def batch_predict(self, data_list: list) -> list:
        """
        Make predictions for multiple equipment readings.

        Args:
            data_list: List of dictionaries containing sensor readings

        Returns:
            List of prediction results
        """
        results = []
        for data in data_list:
            try:
                result = self.predict(data)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'equipment_id': data.get('equipment_id', 'unknown')
                })

        return results


# Initialize service
MODEL_DIR = os.getenv('MODEL_DIR', 'models/checkpoints')
ARTIFACTS_DIR = os.getenv('ARTIFACTS_DIR', 'models/artifacts')

try:
    service = PredictiveMaintenanceService(MODEL_DIR, ARTIFACTS_DIR)
except Exception as e:
    logger.error(f"Failed to initialize service: {e}")
    service = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    if service is None or service.model is None:
        HEALTH_STATUS.set(0)
        return jsonify({'status': 'unhealthy', 'error': 'Model not loaded'}), 503

    HEALTH_STATUS.set(1)
    return jsonify({
        'status': 'healthy',
        'model_version': service.metadata.get('version', 'unknown') if service.metadata else 'unknown',
        'timestamp': pd.Timestamp.now().isoformat()
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint."""
    if service is None:
        return jsonify({'error': 'Service not initialized'}), 503

    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        result = service.predict(data)
        return jsonify(result), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint."""
    if service is None:
        return jsonify({'error': 'Service not initialized'}), 503

    try:
        data = request.get_json()

        if not isinstance(data, list):
            return jsonify({'error': 'Expected list of records'}), 400

        results = service.batch_predict(data)
        return jsonify({'predictions': results, 'count': len(results)}), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    if service is None or service.metadata is None:
        return jsonify({'error': 'Model metadata not available'}), 404

    return jsonify(service.metadata), 200


if __name__ == '__main__':
    # Run the Flask app
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
