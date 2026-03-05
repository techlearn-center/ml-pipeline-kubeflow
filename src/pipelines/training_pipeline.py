"""
training_pipeline.py -- End-to-end KFP v2 training pipeline.

Orchestrates:
  1. data_loader  -- load and split a dataset
  2. trainer      -- train a classification model
  3. evaluator    -- evaluate the model and gate deployment
  4. (conditional) deploy step if evaluation passes thresholds

Compile:
    python -m src.pipelines.training_pipeline

Submit to a running KFP instance:
    python -m src.pipelines.training_pipeline --submit
"""

from __future__ import annotations

import os
from kfp import dsl, compiler

# Import components
from src.components.data_loader import data_loader
from src.components.trainer import trainer
from src.components.evaluator import evaluator


# ---------------------------------------------------------------------------
# Optional: model deployment component (lightweight placeholder)
# ---------------------------------------------------------------------------
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["joblib>=1.3"],
)
def deploy_model(
    model_artifact: dsl.Input[dsl.Model],
    deploy_decision: str,
    model_name: str = "sklearn-classifier",
    serving_endpoint: str = "http://model-server:8501",
) -> str:
    """Copy model to a serving location if deploy_decision == 'deploy'.

    In production this would push the model to a registry (MLflow, Vertex AI
    Model Registry, Seldon, KServe, etc.).  Here we simulate the action.
    """
    import shutil, os, json

    if deploy_decision != "deploy":
        msg = f"Skipping deployment: decision={deploy_decision}"
        print(msg)
        return msg

    deploy_dir = "/tmp/deployed_models"
    os.makedirs(deploy_dir, exist_ok=True)
    dest = os.path.join(deploy_dir, f"{model_name}.joblib")
    shutil.copy2(model_artifact.path, dest)

    manifest = {
        "model_name": model_name,
        "artifact_path": dest,
        "serving_endpoint": serving_endpoint,
        "status": "deployed",
    }
    manifest_path = os.path.join(deploy_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    msg = f"Model '{model_name}' deployed to {dest}"
    print(msg)
    return msg


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------
@dsl.pipeline(
    name="ml-training-pipeline",
    description=(
        "End-to-end ML training pipeline: data loading, preprocessing, "
        "model training, evaluation, and conditional deployment."
    ),
)
def training_pipeline(
    dataset_name: str = "breast_cancer",
    model_type: str = "random_forest",
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    accuracy_threshold: float = 0.85,
    f1_threshold: float = 0.80,
    random_state: int = 42,
    model_name: str = "sklearn-classifier",
) -> None:
    """Define the training pipeline DAG.

    Parameters can be overridden at submission time from the KFP UI or CLI.
    """

    # Step 1 -- Load and split data
    load_task = data_loader(
        dataset_name=dataset_name,
        test_size=test_size,
        random_state=random_state,
    )
    load_task.set_display_name("Load & Split Data")
    load_task.set_caching_options(enable_caching=True)

    # Step 2 -- Train model
    train_task = trainer(
        train_dataset=load_task.outputs["train_dataset"],
        model_type=model_type,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    train_task.set_display_name("Train Model")
    train_task.set_caching_options(enable_caching=True)
    train_task.set_memory_limit("2Gi")
    train_task.set_cpu_limit("2")

    # Step 3 -- Evaluate model
    eval_task = evaluator(
        test_dataset=load_task.outputs["test_dataset"],
        model_artifact=train_task.outputs["model_artifact"],
        accuracy_threshold=accuracy_threshold,
        f1_threshold=f1_threshold,
    )
    eval_task.set_display_name("Evaluate Model")
    eval_task.set_caching_options(enable_caching=False)

    # Step 4 -- Conditionally deploy
    with dsl.If(eval_task.output == "deploy"):
        deploy_task = deploy_model(
            model_artifact=train_task.outputs["model_artifact"],
            deploy_decision=eval_task.output,
            model_name=model_name,
        )
        deploy_task.set_display_name("Deploy Model")


# ---------------------------------------------------------------------------
# Compile / submit
# ---------------------------------------------------------------------------
PIPELINE_YAML = "pipeline.yaml"


def compile_pipeline() -> str:
    """Compile the pipeline to a YAML IR file."""
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=PIPELINE_YAML,
    )
    print(f"Pipeline compiled to {PIPELINE_YAML}")
    return PIPELINE_YAML


def submit_pipeline(endpoint: str | None = None) -> None:
    """Submit the compiled pipeline to a KFP endpoint."""
    from kfp.client import Client

    endpoint = endpoint or os.getenv("KFP_ENDPOINT", "http://localhost:8080")
    client = Client(host=endpoint)

    compile_pipeline()
    run = client.create_run_from_pipeline_package(
        pipeline_file=PIPELINE_YAML,
        arguments={
            "dataset_name": "breast_cancer",
            "model_type": "random_forest",
            "n_estimators": 100,
            "max_depth": 5,
            "accuracy_threshold": 0.85,
            "f1_threshold": 0.80,
        },
        run_name="training-run",
        experiment_name="ml-training-experiment",
    )
    print(f"Pipeline run submitted: {run.run_id}")


if __name__ == "__main__":
    import sys

    if "--submit" in sys.argv:
        submit_pipeline()
    else:
        compile_pipeline()
