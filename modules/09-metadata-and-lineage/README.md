# Module 09: Multi-Step Pipelines -- Train, Evaluate, Deploy

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 08 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Build an end-to-end pipeline that chains training, evaluation, and deployment
- Implement A/B model comparison before promoting a candidate
- Use KServe / Seldon for model serving from a pipeline
- Add notification steps for success and failure
- Create reusable pipeline templates as nested pipelines

---

## Concepts

### The Full ML Lifecycle Pipeline

A production ML pipeline does not stop at training. It covers the entire lifecycle:

```
  +--------+    +-----------+    +--------+    +----------+    +--------+
  | Ingest |--->| Transform |--->| Train  |--->| Evaluate |--->| Deploy |
  +--------+    +-----------+    +--------+    +-----+----+    +---+----+
                                                     |             |
                                                FAIL |        +----v-----+
                                                     v        | Monitor  |
                                               +---------+    +----------+
                                               | Retrain |
                                               +---------+
```

### Multi-Model Comparison Pattern

In production you often train multiple candidates and compare them:

```python
@dsl.pipeline(name="model-comparison-pipeline")
def model_comparison_pipeline(dataset_name: str = "breast_cancer") -> None:
    load_task = data_loader(dataset_name=dataset_name)

    # Train multiple models in parallel
    rf_task = train_model(
        train_data=load_task.outputs["train_data"],
        algorithm="random_forest",
    )
    xgb_task = train_model(
        train_data=load_task.outputs["train_data"],
        algorithm="xgboost",
    )

    # Evaluate both on same test set
    rf_eval = evaluate_model(
        test_data=load_task.outputs["test_data"],
        model=rf_task.outputs["model"],
    )
    xgb_eval = evaluate_model(
        test_data=load_task.outputs["test_data"],
        model=xgb_task.outputs["model"],
    )

    # Pick winner
    selector = select_best_model(
        rf_accuracy=rf_eval.outputs["accuracy"],
        xgb_accuracy=xgb_eval.outputs["accuracy"],
        rf_model=rf_task.outputs["model"],
        xgb_model=xgb_task.outputs["model"],
    )

    # Deploy winner
    with dsl.If(selector.outputs["decision"] == "deploy"):
        deploy_model(model=selector.outputs["best_model"])
```

---

## Hands-On Lab

### Exercise 1: Complete Train-Evaluate-Deploy Pipeline

```python
from kfp import dsl, compiler
from kfp.dsl import Dataset, Model, Metrics, Input, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1", "scikit-learn>=1.3"],
)
def load_and_split(
    dataset_name: str,
    test_size: float,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    stats: Output[Metrics],
) -> None:
    import pandas as pd
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine
    from sklearn.model_selection import train_test_split

    loaders = {
        "breast_cancer": load_breast_cancer,
        "iris": load_iris,
        "wine": load_wine,
    }
    bunch = loaders[dataset_name]()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["target"],
    )
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)

    stats.log_metric("total_samples", len(df))
    stats.log_metric("train_samples", len(train_df))
    stats.log_metric("test_samples", len(test_df))
    print(f"Split complete: {len(train_df)} train, {len(test_df)} test")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.1", "scikit-learn>=1.3",
        "xgboost>=2.0", "joblib>=1.3",
    ],
)
def train_candidate(
    train_data: Input[Dataset],
    model_out: Output[Model],
    metrics_out: Output[Metrics],
    algorithm: str = "random_forest",
    n_estimators: int = 100,
    max_depth: int = 5,
) -> float:
    import pandas as pd, joblib, time
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    df = pd.read_csv(train_data.path)
    X, y = df.drop(columns=["target"]), df["target"]

    estimators = {
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42,
        ),
    }
    if algorithm == "xgboost":
        from xgboost import XGBClassifier
        est = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=42, eval_metric="logloss",
        )
    else:
        est = estimators[algorithm]

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
    start = time.time()
    pipe.fit(X, y)
    elapsed = time.time() - start

    acc = pipe.score(X, y)
    joblib.dump(pipe, model_out.path)

    model_out.metadata["algorithm"] = algorithm
    metrics_out.log_metric("algorithm", algorithm)
    metrics_out.log_metric("train_accuracy", round(acc, 4))
    metrics_out.log_metric("training_seconds", round(elapsed, 2))

    print(f"{algorithm}: accuracy={acc:.4f} ({elapsed:.1f}s)")
    return acc


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.1", "scikit-learn>=1.3",
        "numpy>=1.25", "joblib>=1.3",
    ],
)
def evaluate_and_gate(
    test_data: Input[Dataset],
    model_artifact: Input[Model],
    eval_metrics: Output[Metrics],
    accuracy_threshold: float = 0.85,
    f1_threshold: float = 0.80,
) -> str:
    import pandas as pd, numpy as np, joblib
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    df = pd.read_csv(test_data.path)
    X, y = df.drop(columns=["target"]), df["target"]
    model = joblib.load(model_artifact.path)

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")

    eval_metrics.log_metric("test_accuracy", round(acc, 4))
    eval_metrics.log_metric("test_f1", round(f1, 4))

    print(classification_report(y, y_pred))

    if acc >= accuracy_threshold and f1 >= f1_threshold:
        print(f"DEPLOY: acc={acc:.4f}, f1={f1:.4f}")
        return "deploy"
    else:
        print(f"NO DEPLOY: acc={acc:.4f}, f1={f1:.4f}")
        return "no_deploy"


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["joblib>=1.3"],
)
def deploy_to_serving(
    model_artifact: dsl.Input[dsl.Model],
    model_name: str = "production-model",
) -> str:
    """Simulate deployment to a model serving endpoint."""
    import shutil, os, json
    from datetime import datetime

    deploy_dir = "/tmp/serving"
    os.makedirs(deploy_dir, exist_ok=True)
    dest = os.path.join(deploy_dir, f"{model_name}.joblib")
    shutil.copy2(model_artifact.path, dest)

    manifest = {
        "model_name": model_name,
        "deployed_at": datetime.utcnow().isoformat(),
        "artifact_path": dest,
        "algorithm": model_artifact.metadata.get("algorithm", "unknown"),
    }
    with open(os.path.join(deploy_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    msg = f"Deployed '{model_name}' at {manifest['deployed_at']}"
    print(msg)
    return msg


@dsl.component(base_image="python:3.11-slim")
def notify(message: str, channel: str = "slack") -> None:
    """Send a notification (simulated)."""
    print(f"[{channel.upper()}] {message}")


# -----------------------------------------------------------------------
# Pipeline: Train -> Evaluate -> Conditionally Deploy -> Notify
# -----------------------------------------------------------------------
@dsl.pipeline(
    name="train-evaluate-deploy-pipeline",
    description="End-to-end ML pipeline with conditional deployment and notifications.",
)
def train_evaluate_deploy_pipeline(
    dataset_name: str = "breast_cancer",
    algorithm: str = "random_forest",
    n_estimators: int = 100,
    max_depth: int = 5,
    test_size: float = 0.2,
    accuracy_threshold: float = 0.85,
    f1_threshold: float = 0.80,
    model_name: str = "production-classifier",
) -> None:
    # Step 1: Load data
    load_task = load_and_split(
        dataset_name=dataset_name, test_size=test_size,
    )
    load_task.set_display_name("Load & Split Data")

    # Step 2: Train
    train_task = train_candidate(
        train_data=load_task.outputs["train_data"],
        algorithm=algorithm,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    train_task.set_display_name("Train Candidate Model")
    train_task.set_memory_limit("2Gi")
    train_task.set_cpu_limit("2")

    # Step 3: Evaluate
    eval_task = evaluate_and_gate(
        test_data=load_task.outputs["test_data"],
        model_artifact=train_task.outputs["model_out"],
        accuracy_threshold=accuracy_threshold,
        f1_threshold=f1_threshold,
    )
    eval_task.set_display_name("Evaluate & Gate")

    # Step 4a: Deploy if approved
    with dsl.If(eval_task.output == "deploy"):
        deploy_task = deploy_to_serving(
            model_artifact=train_task.outputs["model_out"],
            model_name=model_name,
        )
        deploy_task.set_display_name("Deploy to Serving")

        success_notify = notify(
            message=f"Model '{model_name}' deployed successfully",
            channel="slack",
        )
        success_notify.set_display_name("Notify: Success")

    # Step 4b: Alert if rejected
    with dsl.Else():
        fail_notify = notify(
            message=f"Model '{model_name}' rejected -- below threshold",
            channel="pagerduty",
        )
        fail_notify.set_display_name("Notify: Rejected")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=train_evaluate_deploy_pipeline,
        package_path="train_evaluate_deploy_pipeline.yaml",
    )
    print("Compiled train_evaluate_deploy_pipeline.yaml")
```

### Exercise 2: Nested Reusable Sub-Pipelines

```python
from kfp import dsl


# Define a reusable training sub-pipeline
@dsl.pipeline(name="training-sub-pipeline")
def training_sub_pipeline(
    train_data: dsl.Input[dsl.Dataset],
    algorithm: str = "random_forest",
    n_estimators: int = 100,
) -> None:
    """Reusable sub-pipeline for training + evaluation."""
    train_task = train_candidate(
        train_data=train_data,
        algorithm=algorithm,
        n_estimators=n_estimators,
    )
    # Evaluation would follow here


# Use the sub-pipeline in a larger pipeline
@dsl.pipeline(name="multi-model-with-subpipelines")
def multi_model_pipeline(dataset_name: str = "breast_cancer") -> None:
    load_task = load_and_split(dataset_name=dataset_name, test_size=0.2)

    # Invoke the sub-pipeline for each algorithm
    rf_sub = training_sub_pipeline(
        train_data=load_task.outputs["train_data"],
        algorithm="random_forest",
    )
    xgb_sub = training_sub_pipeline(
        train_data=load_task.outputs["train_data"],
        algorithm="xgboost",
    )
```

### Exercise 3: KServe Deployment Component

```python
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["kubernetes>=28.0", "pyyaml>=6.0"],
)
def deploy_to_kserve(
    model_uri: str,
    model_name: str = "sklearn-classifier",
    namespace: str = "default",
    min_replicas: int = 1,
    max_replicas: int = 3,
) -> str:
    """Deploy a model to KServe InferenceService."""
    from kubernetes import client, config
    import yaml

    inference_service = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": model_name,
            "namespace": namespace,
        },
        "spec": {
            "predictor": {
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "sklearn": {
                    "storageUri": model_uri,
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "256Mi"},
                        "limits": {"cpu": "1", "memory": "1Gi"},
                    },
                },
            },
        },
    }

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    api = client.CustomObjectsApi()

    try:
        api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            body=inference_service,
        )
        msg = f"Created InferenceService '{model_name}' in namespace '{namespace}'"
    except client.ApiException as e:
        if e.status == 409:
            api.patch_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=model_name,
                body=inference_service,
            )
            msg = f"Updated InferenceService '{model_name}'"
        else:
            raise

    print(msg)
    return msg
```

---

## Pipeline Design Patterns

| Pattern | Description |
|---|---|
| **Train-Evaluate-Deploy** | Linear pipeline with a gate between eval and deploy |
| **Champion-Challenger** | Compare new model to current production model |
| **Multi-Model Tournament** | Train N candidates, evaluate all, deploy the best |
| **Blue-Green Deploy** | Deploy new model alongside old, shift traffic gradually |
| **Canary Deploy** | Route a small fraction of traffic to the new model first |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| No notification on failure | Team is unaware of broken pipelines | Add notification steps in both If and Else branches |
| Deploying without evaluation | Bad model in production | Always include an evaluation gate |
| Single model training | Missed better algorithms | Train and compare multiple candidates |
| No rollback plan | Cannot revert a bad deployment | Keep previous model version; implement rollback step |

---

## Self-Check Questions

1. Why should deployment be conditional on evaluation results?
2. How does the champion-challenger pattern prevent model regression?
3. What is the advantage of using nested sub-pipelines?
4. How would you implement a rollback step if a newly deployed model performs poorly?
5. What information should a deployment notification include?

---

## You Know You Have Completed This Module When...

- [ ] Built a complete train-evaluate-deploy pipeline
- [ ] Implemented conditional deployment with notifications
- [ ] Understand the champion-challenger pattern
- [ ] Created a reusable sub-pipeline
- [ ] Can explain KServe InferenceService deployment

---

**Next: [Module 10 -- Production Kubeflow Operations -->](../10-production-kubeflow/)**
