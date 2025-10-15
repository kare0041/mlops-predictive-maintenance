"""
Automated retraining trigger for predictive maintenance models.

Monitors drift metrics and model performance to decide when retraining is needed.
"""

import os
import json
from datetime import datetime, timedelta
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingTrigger:
    """
    Monitors model performance and drift to trigger retraining.
    """

    def __init__(self,
                 ml_client: MLClient,
                 min_performance_threshold: float = 0.85,
                 max_drift_proportion: float = 0.3,
                 min_days_since_training: int = 7):
        """
        Initialize retraining trigger.

        Args:
            ml_client: Azure ML client
            min_performance_threshold: Minimum acceptable PR-AUC
            max_drift_proportion: Maximum acceptable drift proportion
            min_days_since_training: Minimum days between retraining
        """
        self.ml_client = ml_client
        self.min_performance_threshold = min_performance_threshold
        self.max_drift_proportion = max_drift_proportion
        self.min_days_since_training = min_days_since_training

    def check_performance_degradation(self, current_metrics_path: str) -> bool:
        """
        Check if model performance has degraded.

        Args:
            current_metrics_path: Path to current evaluation metrics

        Returns:
            True if performance degraded
        """
        try:
            with open(current_metrics_path, 'r') as f:
                metrics = json.load(f)

            pr_auc = metrics.get('pr_auc', 0)

            if pr_auc < self.min_performance_threshold:
                logger.warning(
                    f"Performance degradation detected! "
                    f"PR-AUC: {pr_auc:.4f} < {self.min_performance_threshold}"
                )
                return True

            logger.info(f"Performance OK: PR-AUC = {pr_auc:.4f}")
            return False

        except Exception as e:
            logger.error(f"Error checking performance: {e}")
            return False

    def check_drift(self, drift_report_path: str) -> bool:
        """
        Check if data drift exceeds threshold.

        Args:
            drift_report_path: Path to drift detection report

        Returns:
            True if significant drift detected
        """
        try:
            with open(drift_report_path, 'r') as f:
                drift_report = json.load(f)

            drift_proportion = drift_report['summary']['drift_proportion']
            drift_detected = drift_report['summary']['drift_detected']

            if drift_detected:
                logger.warning(
                    f"Data drift detected! "
                    f"Drift proportion: {drift_proportion:.2%} "
                    f"(threshold: {self.max_drift_proportion:.2%})"
                )
                return True

            logger.info(f"No significant drift: {drift_proportion:.2%}")
            return False

        except Exception as e:
            logger.error(f"Error checking drift: {e}")
            return False

    def check_time_since_training(self, last_training_date: str) -> bool:
        """
        Check if enough time has passed since last training.

        Args:
            last_training_date: ISO format date string

        Returns:
            True if sufficient time has passed
        """
        try:
            last_date = datetime.fromisoformat(last_training_date)
            days_elapsed = (datetime.now() - last_date).days

            if days_elapsed >= self.min_days_since_training:
                logger.info(
                    f"{days_elapsed} days since last training "
                    f"(threshold: {self.min_days_since_training})"
                )
                return True

            logger.info(f"Too soon to retrain: {days_elapsed} days elapsed")
            return False

        except Exception as e:
            logger.error(f"Error checking training date: {e}")
            return True  # Default to allowing retraining

    def should_retrain(self,
                      metrics_path: str = None,
                      drift_path: str = None,
                      last_training_date: str = None) -> dict:
        """
        Determine if model should be retrained.

        Args:
            metrics_path: Path to current metrics
            drift_path: Path to drift report
            last_training_date: Date of last training

        Returns:
            Dictionary with decision and reasons
        """
        reasons = []
        should_retrain = False

        # Check performance
        if metrics_path and os.path.exists(metrics_path):
            if self.check_performance_degradation(metrics_path):
                reasons.append("performance_degradation")
                should_retrain = True

        # Check drift
        if drift_path and os.path.exists(drift_path):
            if self.check_drift(drift_path):
                reasons.append("data_drift")
                should_retrain = True

        # Check time
        if last_training_date:
            if self.check_time_since_training(last_training_date):
                reasons.append("scheduled_retrain")
                # Time alone doesn't trigger, but supports other reasons
                if not should_retrain and len(reasons) == 1:
                    # Only time reason - don't retrain unless other conditions met
                    reasons = []

        decision = {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'timestamp': datetime.now().isoformat(),
            'thresholds': {
                'min_performance': self.min_performance_threshold,
                'max_drift': self.max_drift_proportion,
                'min_days': self.min_days_since_training
            }
        }

        logger.info(f"Retraining decision: {decision}")
        return decision

    def trigger_pipeline(self, pipeline_name: str = "predictive_maintenance_training"):
        """
        Trigger the Azure ML retraining pipeline.

        Args:
            pipeline_name: Name of pipeline to trigger

        Returns:
            Pipeline job
        """
        try:
            from pipeline_definition import submit_pipeline

            logger.info(f"Triggering retraining pipeline: {pipeline_name}")

            data_path = "azureml://datastores/workspaceblobstore/paths/data/sample_sensor_data.csv"
            job = submit_pipeline(self.ml_client, data_path)

            logger.info(f"Pipeline triggered successfully: {job.name}")
            return job

        except Exception as e:
            logger.error(f"Failed to trigger pipeline: {e}")
            raise


def main():
    """Main execution for retraining trigger."""

    # Initialize Azure ML client
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
        workspace_name=os.getenv("AZURE_ML_WORKSPACE"),
    )

    # Initialize trigger
    trigger = RetrainingTrigger(
        ml_client=ml_client,
        min_performance_threshold=0.85,
        max_drift_proportion=0.3,
        min_days_since_training=7
    )

    # Check conditions
    decision = trigger.should_retrain(
        metrics_path="models/evaluation/evaluation_metrics.json",
        drift_path="models/drift_reports/latest_drift_report.json",
        last_training_date="2024-01-01T00:00:00"
    )

    # Save decision
    os.makedirs("models/retrain_decisions", exist_ok=True)
    decision_path = f"models/retrain_decisions/decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(decision_path, 'w') as f:
        json.dump(decision, f, indent=2)

    logger.info(f"Decision saved to {decision_path}")

    # Trigger retraining if needed
    if decision['should_retrain']:
        logger.info("Retraining conditions met - triggering pipeline")
        job = trigger.trigger_pipeline()
        logger.info(f"Retraining job submitted: {job.name}")
    else:
        logger.info("No retraining needed at this time")


if __name__ == "__main__":
    main()
