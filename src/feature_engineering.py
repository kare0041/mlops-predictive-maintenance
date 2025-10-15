"""
Feature engineering module for predictive maintenance.

Creates advanced features including rolling statistics, interactions,
and domain-specific features for equipment failure prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for sensor time-series data.

    Creates rolling statistics, interaction features, and domain-specific
    predictive maintenance features.
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = None
        self.feature_names = None

    def create_rolling_features(self,
                                df: pd.DataFrame,
                                windows: List[int] = [6, 12, 24],
                                sensor_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create rolling window statistics for sensor readings.

        Args:
            df: Input DataFrame
            windows: List of window sizes (in hours)
            sensor_cols: Sensor columns to compute statistics for

        Returns:
            DataFrame with rolling features
        """
        df = df.copy()

        if sensor_cols is None:
            sensor_cols = ['temperature', 'vibration', 'pressure',
                          'rotation_speed', 'current']

        # Sort by timestamp and equipment_id for proper rolling computation
        if 'timestamp' in df.columns and 'equipment_id' in df.columns:
            df = df.sort_values(['equipment_id', 'timestamp'])

        for window in windows:
            for col in sensor_cols:
                if col in df.columns:
                    # Rolling mean
                    df[f'{col}_rolling_mean_{window}h'] = (
                        df.groupby('equipment_id')[col]
                        .transform(lambda x: x.rolling(window, min_periods=1).mean())
                    ) if 'equipment_id' in df.columns else df[col].rolling(window, min_periods=1).mean()

                    # Rolling std
                    df[f'{col}_rolling_std_{window}h'] = (
                        df.groupby('equipment_id')[col]
                        .transform(lambda x: x.rolling(window, min_periods=1).std())
                    ) if 'equipment_id' in df.columns else df[col].rolling(window, min_periods=1).std()

                    # Rolling max
                    df[f'{col}_rolling_max_{window}h'] = (
                        df.groupby('equipment_id')[col]
                        .transform(lambda x: x.rolling(window, min_periods=1).max())
                    ) if 'equipment_id' in df.columns else df[col].rolling(window, min_periods=1).max()

        # Fill any NaN values from rolling operations
        df = df.fillna(method='bfill').fillna(0)

        logger.info(f"Created rolling features for windows: {windows}")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between sensors.

        These capture relationships that may indicate failure modes.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with interaction features
        """
        df = df.copy()

        # Temperature-Vibration interaction (high temp + high vibration = bearing issue)
        if 'temperature' in df.columns and 'vibration' in df.columns:
            df['temp_vibration_interaction'] = df['temperature'] * df['vibration']

        # Pressure-Speed interaction (pressure drop at high speed = pump issue)
        if 'pressure' in df.columns and 'rotation_speed' in df.columns:
            df['pressure_speed_ratio'] = df['pressure'] / (df['rotation_speed'] + 1)

        # Current-Speed interaction (high current with low speed = motor issue)
        if 'current' in df.columns and 'rotation_speed' in df.columns:
            df['current_speed_ratio'] = df['current'] / (df['rotation_speed'] + 1)

        # Temperature efficiency (temp vs expected based on current)
        if 'temperature' in df.columns and 'current' in df.columns:
            df['temp_current_ratio'] = df['temperature'] / (df['current'] + 1)

        logger.info("Created interaction features")
        return df

    def create_degradation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that capture equipment degradation over time.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with degradation features
        """
        df = df.copy()

        sensor_cols = ['temperature', 'vibration', 'pressure',
                      'rotation_speed', 'current']

        for col in sensor_cols:
            if col in df.columns:
                # Rate of change (first derivative approximation)
                if 'equipment_id' in df.columns:
                    df[f'{col}_rate_of_change'] = (
                        df.groupby('equipment_id')[col]
                        .transform(lambda x: x.diff())
                    )
                else:
                    df[f'{col}_rate_of_change'] = df[col].diff()

                # Deviation from running average
                if 'equipment_id' in df.columns:
                    running_mean = (
                        df.groupby('equipment_id')[col]
                        .transform(lambda x: x.expanding().mean())
                    )
                else:
                    running_mean = df[col].expanding().mean()

                df[f'{col}_deviation_from_avg'] = df[col] - running_mean

        # Fill NaN values
        df = df.fillna(0)

        logger.info("Created degradation features")
        return df

    def create_health_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite health indicator features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with health indicators
        """
        df = df.copy()

        # Overall vibration health score (lower is better)
        if 'vibration' in df.columns:
            vibration_threshold = 2.0  # mm/s
            df['vibration_health_score'] = np.clip(
                1 - (df['vibration'] / vibration_threshold), 0, 1
            )

        # Temperature health score
        if 'temperature' in df.columns:
            temp_optimal = 65.0
            temp_critical = 95.0
            df['temperature_health_score'] = np.clip(
                1 - (np.abs(df['temperature'] - temp_optimal) / (temp_critical - temp_optimal)),
                0, 1
            )

        # Composite equipment health index
        health_cols = [col for col in df.columns if 'health_score' in col]
        if health_cols:
            df['composite_health_index'] = df[health_cols].mean(axis=1)

        logger.info("Created health indicator features")
        return df

    def scale_features(self,
                      df: pd.DataFrame,
                      fit: bool = True,
                      method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.

        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for inference)
            method: Scaling method ('standard' or 'robust')

        Returns:
            DataFrame with scaled features
        """
        df = df.copy()

        # Get numerical columns (exclude target and identifiers)
        exclude_cols = ['failure', 'equipment_id', 'timestamp', 'cycle']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        scale_cols = [col for col in numeric_cols if col not in exclude_cols]

        if fit:
            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")

            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
            logger.info(f"Fitted {method} scaler on {len(scale_cols)} features")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df[scale_cols] = self.scaler.transform(df[scale_cols])
            logger.info(f"Applied {method} scaler to features")

        return df

    def select_features(self,
                       df: pd.DataFrame,
                       target_col: str = 'failure') -> tuple:
        """
        Separate features and target variable.

        Args:
            df: Input DataFrame
            target_col: Name of target column

        Returns:
            Tuple of (X, y, feature_names)
        """
        # Drop non-feature columns
        drop_cols = [target_col, 'timestamp', 'equipment_id']
        feature_cols = [col for col in df.columns if col not in drop_cols]

        X = df[feature_cols].values
        y = df[target_col].values

        self.feature_names = feature_cols

        logger.info(f"Selected {len(feature_cols)} features for modeling")
        return X, y, feature_cols

    def save_artifacts(self, output_dir: str):
        """
        Save feature engineering artifacts (scaler, feature names).

        Args:
            output_dir: Directory to save artifacts
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        if self.scaler is not None:
            scaler_path = os.path.join(output_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")

        if self.feature_names is not None:
            features_path = os.path.join(output_dir, 'feature_names.pkl')
            joblib.dump(self.feature_names, features_path)
            logger.info(f"Saved feature names to {features_path}")

    def load_artifacts(self, input_dir: str):
        """
        Load feature engineering artifacts.

        Args:
            input_dir: Directory containing artifacts
        """
        import os

        scaler_path = os.path.join(input_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")

        features_path = os.path.join(input_dir, 'feature_names.pkl')
        if os.path.exists(features_path):
            self.feature_names = joblib.load(features_path)
            logger.info(f"Loaded feature names from {features_path}")


def engineer_features_pipeline(input_path: str,
                               output_path: str,
                               is_training: bool = True,
                               artifacts_dir: str = 'models/artifacts/') -> str:
    """
    Complete feature engineering pipeline.

    Args:
        input_path: Path to preprocessed data
        output_path: Path to save engineered features
        is_training: Whether this is training data (fits scaler)
        artifacts_dir: Directory to save/load artifacts

    Returns:
        Path to output file
    """
    import os

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Load artifacts if not training
    if not is_training and os.path.exists(artifacts_dir):
        fe.load_artifacts(artifacts_dir)

    # Load data
    df = pd.read_csv(input_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    logger.info(f"Loaded {len(df)} samples from {input_path}")

    # Create features
    df = fe.create_rolling_features(df)
    df = fe.create_interaction_features(df)
    df = fe.create_degradation_features(df)
    df = fe.create_health_indicators(df)

    # Scale features
    df = fe.scale_features(df, fit=is_training, method='robust')

    # Save artifacts if training
    if is_training:
        fe.save_artifacts(artifacts_dir)

    # Save engineered features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved engineered features to {output_path}")

    return output_path


if __name__ == "__main__":
    # Example usage
    train_path = engineer_features_pipeline(
        input_path='data/processed/train.csv',
        output_path='data/processed/train_features.csv',
        is_training=True
    )

    test_path = engineer_features_pipeline(
        input_path='data/processed/test.csv',
        output_path='data/processed/test_features.csv',
        is_training=False
    )

    print(f"\nFeature engineering complete!")
    print(f"Training features: {train_path}")
    print(f"Testing features: {test_path}")
