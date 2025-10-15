"""
Data preprocessing module for predictive maintenance pipeline.

Handles data loading, cleaning, validation, and train/test splitting.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for sensor data in predictive maintenance tasks.

    Handles missing values, outliers, and data validation.
    """

    def __init__(self,
                 missing_threshold: float = 0.3,
                 outlier_std: float = 5.0):
        """
        Initialize the preprocessor.

        Args:
            missing_threshold: Maximum proportion of missing values allowed per column
            outlier_std: Number of standard deviations for outlier detection
        """
        self.missing_threshold = missing_threshold
        self.outlier_std = outlier_std
        self.feature_means = {}
        self.feature_stds = {}

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load sensor data from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        return df

    def handle_missing_values(self,
                             df: pd.DataFrame,
                             method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame
            method: Imputation method ('forward_fill', 'mean', 'drop')

        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()

        # Check missing value proportion
        missing_props = df.isnull().sum() / len(df)
        high_missing = missing_props[missing_props > self.missing_threshold]

        if len(high_missing) > 0:
            logger.warning(f"Columns with >{self.missing_threshold:.0%} missing: {high_missing.index.tolist()}")

        # Apply imputation strategy
        if method == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown imputation method: {method}")

        logger.info(f"Missing values handled using '{method}' method")
        return df

    def remove_outliers(self,
                       df: pd.DataFrame,
                       columns: Optional[list] = None) -> pd.DataFrame:
        """
        Remove outliers using z-score method.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers (default: all numeric)

        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude label column
            columns = [c for c in columns if c != 'failure']

        initial_len = len(df)

        for col in columns:
            if col in df.columns:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < self.outlier_std]

        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} outlier samples ({removed/initial_len:.2%})")

        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp.

        Args:
            df: Input DataFrame with 'timestamp' column

        Returns:
            DataFrame with additional time features
        """
        df = df.copy()

        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

            logger.info("Created time-based features: hour, day_of_week, month, is_weekend")

        return df

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data quality and schema.

        Args:
            df: DataFrame to validate

        Returns:
            True if validation passes
        """
        required_columns = ['temperature', 'vibration', 'pressure',
                          'rotation_speed', 'current', 'failure']

        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False

        # Check for negative values in sensor readings
        sensor_cols = ['temperature', 'vibration', 'pressure',
                      'rotation_speed', 'current']
        for col in sensor_cols:
            if (df[col] < 0).any():
                logger.warning(f"Negative values found in {col}")

        # Check label distribution
        failure_rate = df['failure'].mean()
        logger.info(f"Failure rate: {failure_rate:.2%}")

        if failure_rate < 0.01:
            logger.warning("Very low failure rate - model may struggle to learn")
        elif failure_rate > 0.5:
            logger.warning("High failure rate - check data quality")

        logger.info("Data validation passed")
        return True

    def prepare_train_test_split(self,
                                 df: pd.DataFrame,
                                 test_size: float = 0.2,
                                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.

        Args:
            df: Input DataFrame
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_df, test_df)
        """
        # Stratified split to maintain failure rate
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['failure'],
            random_state=random_state
        )

        logger.info(f"Train set: {len(train_df)} samples ({1-test_size:.0%})")
        logger.info(f"Test set: {len(test_df)} samples ({test_size:.0%})")
        logger.info(f"Train failure rate: {train_df['failure'].mean():.2%}")
        logger.info(f"Test failure rate: {test_df['failure'].mean():.2%}")

        return train_df, test_df


def preprocess_pipeline(input_path: str,
                       output_dir: str = 'data/processed/',
                       test_size: float = 0.2) -> Tuple[str, str]:
    """
    Complete preprocessing pipeline.

    Args:
        input_path: Path to raw data CSV
        output_dir: Directory to save processed data
        test_size: Proportion for test set

    Returns:
        Tuple of (train_path, test_path)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Load data
    df = preprocessor.load_data(input_path)

    # Validate
    preprocessor.validate_data(df)

    # Handle missing values
    df = preprocessor.handle_missing_values(df, method='forward_fill')

    # Create time features
    df = preprocessor.create_time_features(df)

    # Remove outliers
    df = preprocessor.remove_outliers(df)

    # Train/test split
    train_df, test_df = preprocessor.prepare_train_test_split(df, test_size=test_size)

    # Save processed data
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved training data to {train_path}")
    logger.info(f"Saved testing data to {test_path}")

    return train_path, test_path


if __name__ == "__main__":
    # Example usage
    train_path, test_path = preprocess_pipeline(
        input_path='data/sample_sensor_data.csv',
        output_dir='data/processed/'
    )
    print(f"\nPreprocessing complete!")
    print(f"Training data: {train_path}")
    print(f"Testing data: {test_path}")
