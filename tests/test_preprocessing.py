"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample sensor data for testing."""
    np.random.seed(42)
    n_samples = 100

    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'equipment_id': np.random.choice(['EQ001', 'EQ002'], n_samples),
        'temperature': np.random.normal(70, 10, n_samples),
        'vibration': np.random.normal(0.5, 0.2, n_samples),
        'pressure': np.random.normal(100, 5, n_samples),
        'rotation_speed': np.random.normal(1500, 100, n_samples),
        'current': np.random.normal(15, 2, n_samples),
        'failure': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }

    return pd.DataFrame(data)


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(missing_threshold=0.3, outlier_std=5.0)
        assert preprocessor.missing_threshold == 0.3
        assert preprocessor.outlier_std == 5.0

    def test_handle_missing_values_forward_fill(self, sample_data):
        """Test forward fill missing value imputation."""
        preprocessor = DataPreprocessor()

        # Introduce missing values
        sample_data.loc[5:10, 'temperature'] = np.nan

        result = preprocessor.handle_missing_values(sample_data, method='forward_fill')

        assert result['temperature'].isna().sum() == 0

    def test_handle_missing_values_mean(self, sample_data):
        """Test mean imputation."""
        preprocessor = DataPreprocessor()

        # Introduce missing values
        sample_data.loc[5:10, 'temperature'] = np.nan
        original_mean = sample_data['temperature'].mean()

        result = preprocessor.handle_missing_values(sample_data, method='mean')

        assert result['temperature'].isna().sum() == 0
        np.testing.assert_almost_equal(result['temperature'].mean(), original_mean, decimal=1)

    def test_create_time_features(self, sample_data):
        """Test time-based feature creation."""
        preprocessor = DataPreprocessor()

        result = preprocessor.create_time_features(sample_data)

        assert 'hour' in result.columns
        assert 'day_of_week' in result.columns
        assert 'month' in result.columns
        assert 'is_weekend' in result.columns

        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23

    def test_validate_data_success(self, sample_data):
        """Test data validation passes with valid data."""
        preprocessor = DataPreprocessor()

        is_valid = preprocessor.validate_data(sample_data)

        assert is_valid is True

    def test_validate_data_missing_columns(self, sample_data):
        """Test data validation fails with missing required columns."""
        preprocessor = DataPreprocessor()

        # Remove required column
        df = sample_data.drop(columns=['temperature'])

        is_valid = preprocessor.validate_data(df)

        assert is_valid is False

    def test_prepare_train_test_split(self, sample_data):
        """Test train/test split maintains failure rate."""
        preprocessor = DataPreprocessor()

        train_df, test_df = preprocessor.prepare_train_test_split(
            sample_data,
            test_size=0.2,
            random_state=42
        )

        # Check sizes
        assert len(train_df) == 80
        assert len(test_df) == 20

        # Check stratification (failure rates should be similar)
        train_failure_rate = train_df['failure'].mean()
        test_failure_rate = test_df['failure'].mean()
        overall_failure_rate = sample_data['failure'].mean()

        np.testing.assert_almost_equal(train_failure_rate, overall_failure_rate, decimal=1)
        np.testing.assert_almost_equal(test_failure_rate, overall_failure_rate, decimal=1)

    def test_remove_outliers(self, sample_data):
        """Test outlier removal."""
        preprocessor = DataPreprocessor(outlier_std=3.0)

        # Add extreme outliers
        sample_data.loc[0, 'temperature'] = 1000  # Extreme value

        result = preprocessor.remove_outliers(sample_data)

        # Should have fewer rows after outlier removal
        assert len(result) < len(sample_data)
