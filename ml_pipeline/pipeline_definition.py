"""
Azure ML Pipeline definition for predictive maintenance.

Orchestrates data preprocessing, feature engineering, model training,
evaluation, and deployment in an automated workflow.
"""

import os
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import command
from azure.ai.ml.entities import Environment, AmlCompute
from azure.identity import DefaultAzureCredential
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Azure ML Configuration
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "your-subscription-id")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "rg-predictive-maintenance")
WORKSPACE_NAME = os.getenv("AZURE_ML_WORKSPACE", "mlw-predictive-maintenance")


def create_compute_cluster(ml_client: MLClient, compute_name: str = "cpu-cluster"):
    """
    Create or get existing compute cluster.

    Args:
        ml_client: Azure ML client
        compute_name: Name of compute cluster

    Returns:
        Compute cluster
    """
    try:
        compute = ml_client.compute.get(compute_name)
        logger.info(f"Using existing compute cluster: {compute_name}")
    except Exception:
        logger.info(f"Creating new compute cluster: {compute_name}")
        compute = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size="STANDARD_DS3_V2",
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=120,
        )
        ml_client.compute.begin_create_or_update(compute).result()

    return compute


def create_environment(ml_client: MLClient):
    """
    Create custom environment with required dependencies.

    Args:
        ml_client: Azure ML client

    Returns:
        Environment object
    """
    env = Environment(
        name="predictive-maintenance-env",
        description="Environment for predictive maintenance pipeline",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file="ml_pipeline/conda_env.yml",
    )

    ml_client.environments.create_or_update(env)
    logger.info("Created/updated environment")
    return env


@pipeline(
    name="predictive_maintenance_training",
    description="End-to-end predictive maintenance training pipeline",
    default_compute="cpu-cluster",
)
def predictive_maintenance_pipeline(
    raw_data: Input,
    test_size: float = 0.2,
    epochs: int = 50,
    batch_size: int = 128,
):
    """
    Define the Azure ML pipeline.

    Args:
        raw_data: Input raw sensor data
        test_size: Test set proportion
        epochs: Training epochs
        batch_size: Batch size

    Returns:
        Pipeline with all components
    """

    # Step 1: Data Preprocessing
    preprocessing_step = command(
        name="data_preprocessing",
        display_name="Data Preprocessing",
        code="../src",
        command="python data_preprocessing.py --input ${{inputs.raw_data}} --output ${{outputs.processed_data}} --test-size ${{inputs.test_size}}",
        environment="predictive-maintenance-env",
        inputs={
            "raw_data": raw_data,
            "test_size": test_size,
        },
        outputs={
            "processed_data": Output(type="uri_folder"),
        },
    )

    # Step 2: Feature Engineering
    feature_engineering_step = command(
        name="feature_engineering",
        display_name="Feature Engineering",
        code="../src",
        command="python feature_engineering.py --input ${{inputs.processed_data}} --output ${{outputs.features}}",
        environment="predictive-maintenance-env",
        inputs={
            "processed_data": preprocessing_step.outputs.processed_data,
        },
        outputs={
            "features": Output(type="uri_folder"),
        },
    )

    # Step 3: Model Training
    training_step = command(
        name="model_training",
        display_name="Model Training",
        code="../src",
        command="python train_model.py --train-data ${{inputs.train_features}}/train_features.csv --test-data ${{inputs.train_features}}/test_features.csv --output-dir ${{outputs.model}} --epochs ${{inputs.epochs}} --batch-size ${{inputs.batch_size}}",
        environment="predictive-maintenance-env",
        inputs={
            "train_features": feature_engineering_step.outputs.features,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        outputs={
            "model": Output(type="uri_folder"),
        },
    )

    # Step 4: Model Evaluation
    evaluation_step = command(
        name="model_evaluation",
        display_name="Model Evaluation",
        code="../src",
        command="python evaluate_model.py --model-path ${{inputs.model}}/model.keras --test-data ${{inputs.test_features}}/test_features.csv --output-dir ${{outputs.evaluation}}",
        environment="predictive-maintenance-env",
        inputs={
            "model": training_step.outputs.model,
            "test_features": feature_engineering_step.outputs.features,
        },
        outputs={
            "evaluation": Output(type="uri_folder"),
        },
    )

    # Step 5: Model Registration (conditional on performance)
    registration_step = command(
        name="model_registration",
        display_name="Model Registration",
        code=".",
        command="python register_model.py --model-path ${{inputs.model}} --metrics-path ${{inputs.evaluation}}/evaluation_metrics.json",
        environment="predictive-maintenance-env",
        inputs={
            "model": training_step.outputs.model,
            "evaluation": evaluation_step.outputs.evaluation,
        },
    )

    return {
        "processed_data": preprocessing_step.outputs.processed_data,
        "features": feature_engineering_step.outputs.features,
        "model": training_step.outputs.model,
        "evaluation": evaluation_step.outputs.evaluation,
    }


def submit_pipeline(ml_client: MLClient, data_path: str):
    """
    Submit the pipeline for execution.

    Args:
        ml_client: Azure ML client
        data_path: Path to raw data

    Returns:
        Pipeline job
    """
    # Create pipeline
    pipeline_job = predictive_maintenance_pipeline(
        raw_data=Input(type="uri_file", path=data_path),
        test_size=0.2,
        epochs=50,
        batch_size=128,
    )

    # Submit
    logger.info("Submitting pipeline job...")
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="predictive_maintenance"
    )

    logger.info(f"Pipeline submitted: {pipeline_job.name}")
    logger.info(f"Job URL: {pipeline_job.studio_url}")

    return pipeline_job


if __name__ == "__main__":
    # Initialize Azure ML client
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    # Create compute and environment
    create_compute_cluster(ml_client)
    create_environment(ml_client)

    # Submit pipeline
    data_path = "azureml://datastores/workspaceblobstore/paths/data/sample_sensor_data.csv"
    job = submit_pipeline(ml_client, data_path)

    logger.info("Pipeline setup complete!")
