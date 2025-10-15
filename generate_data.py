"""
Generate synthetic sensor data for predictive maintenance demonstration.

This script creates realistic time-series equipment sensor data with:
- Multiple sensor readings (temperature, vibration, pressure, etc.)
- Temporal patterns and degradation trends
- Equipment failure events
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)


def generate_sensor_data(n_samples=10000):
    """
    Generate synthetic sensor data for predictive maintenance.

    Args:
        n_samples: Number of time-series samples to generate

    Returns:
        DataFrame with sensor readings and failure labels
    """
    # Generate timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_samples)]

    # Initialize equipment cycles (used to simulate degradation)
    cycle = np.arange(n_samples)

    # Simulate failure events (approximately 5% failure rate)
    failure_probability = 0.05
    failures = np.zeros(n_samples)

    # Create failure windows - equipment degrades before failure
    failure_indices = np.random.choice(
        range(100, n_samples - 100),
        size=int(n_samples * failure_probability),
        replace=False
    )

    for idx in failure_indices:
        # Mark failure point and degradation window (48 hours before)
        failures[max(0, idx - 48):idx + 1] = 1

    # Generate sensor readings with realistic patterns
    data = {
        'timestamp': timestamps,
        'equipment_id': np.random.choice(['EQ001', 'EQ002', 'EQ003', 'EQ004'], n_samples),
        'cycle': cycle,
    }

    # Temperature sensor (°C) - increases before failure
    base_temp = 65.0
    temp_noise = np.random.normal(0, 2, n_samples)
    temp_trend = failures * np.random.uniform(10, 25, n_samples)
    data['temperature'] = base_temp + temp_noise + temp_trend

    # Vibration sensor (mm/s) - increases significantly before failure
    base_vibration = 0.5
    vibration_noise = np.random.normal(0, 0.1, n_samples)
    vibration_trend = failures * np.random.uniform(1.5, 3.0, n_samples)
    data['vibration'] = np.maximum(0, base_vibration + vibration_noise + vibration_trend)

    # Pressure sensor (bar) - fluctuates before failure
    base_pressure = 100.0
    pressure_noise = np.random.normal(0, 5, n_samples)
    pressure_trend = failures * np.random.uniform(-15, -5, n_samples)
    data['pressure'] = base_pressure + pressure_noise + pressure_trend

    # Rotation speed (RPM) - decreases before failure
    base_rpm = 1500
    rpm_noise = np.random.normal(0, 50, n_samples)
    rpm_trend = failures * np.random.uniform(-200, -50, n_samples)
    data['rotation_speed'] = np.maximum(0, base_rpm + rpm_noise + rpm_trend)

    # Current draw (Amperes) - increases before failure
    base_current = 15.0
    current_noise = np.random.normal(0, 1, n_samples)
    current_trend = failures * np.random.uniform(5, 15, n_samples)
    data['current'] = base_current + current_noise + current_trend

    # Operating hours (cumulative)
    data['operating_hours'] = cycle * 0.5 + np.random.normal(0, 10, n_samples)

    # Add failure label (1 = will fail within next 48 hours, 0 = normal)
    data['failure'] = failures.astype(int)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add some realistic missing values (1-2%)
    missing_mask = np.random.random(df.shape) < 0.015
    df = df.mask(missing_mask)

    return df


if __name__ == "__main__":
    print("Generating synthetic sensor data...")

    # Generate training dataset (10,000 samples)
    df = generate_sensor_data(n_samples=10000)

    # Save to CSV
    output_path = "data/sample_sensor_data.csv"
    df.to_csv(output_path, index=False)

    print(f"[OK] Generated {len(df)} samples")
    print(f"[OK] Saved to {output_path}")
    print(f"\nDataset summary:")
    print(f"  - Failure rate: {df['failure'].mean():.2%}")
    print(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  - Equipment IDs: {df['equipment_id'].nunique()}")
    print(f"\nSensor statistics:")
    print(df.describe())
