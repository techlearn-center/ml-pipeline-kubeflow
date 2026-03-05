# Module 08: Metadata Tracking and Lineage

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 07 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain how ML Metadata (MLMD) tracks artifacts, executions, and contexts
- Query the MLMD store to trace a model back to its training data
- Add custom metadata properties to component outputs
- Use MLflow alongside KFP for experiment tracking
- Build a lineage-aware pipeline that records provenance at every step

---

## Concepts

### What Is ML Metadata (MLMD)?

ML Metadata is the backbone of Kubeflow Pipelines' tracking system. Every time a pipeline runs, MLMD automatically records:

| Entity | What it represents | Example |
|---|---|---|
| **Artifact** | A versioned data object | A CSV dataset, a trained model, a metrics JSON |
| **Execution** | A single run of a component | "data_loader executed at 2025-01-15T02:00Z" |
| **Context** | A grouping of executions and artifacts | A pipeline run, an experiment |
| **Event** | An input/output relationship | "Execution E1 produced Artifact A1" |

```
  Context (Pipeline Run "run-42")
  +-------------------------------------------------+
  |                                                 |
  |  Execution: data_loader                         |
  |    Input:  (parameters)                         |
  |    Output: Artifact "train.csv" (id=101)        |
  |            Artifact "test.csv"  (id=102)        |
  |                                                 |
  |  Execution: trainer                             |
  |    Input:  Artifact "train.csv" (id=101)        |
  |    Output: Artifact "model.pkl" (id=201)        |
  |                                                 |
  |  Execution: evaluator                           |
  |    Input:  Artifact "test.csv"  (id=102)        |
  |            Artifact "model.pkl" (id=201)        |
  |    Output: Artifact "metrics"   (id=301)        |
  +-------------------------------------------------+
```

### Lineage Queries

Lineage lets you answer questions like:

- **Forward lineage:** "Which models were trained on dataset X?"
- **Backward lineage:** "What data was used to train model Y?"
- **Impact analysis:** "If we discover a bug in the preprocessing step, which models are affected?"

### Adding Custom Metadata

Every KFP artifact can carry custom metadata via the `.metadata` dictionary:

```python
@dsl.component(base_image="python:3.11-slim")
def my_component(
    model_out: dsl.Output[dsl.Model],
) -> None:
    # Custom metadata is persisted in MLMD
    model_out.metadata["algorithm"] = "random_forest"
    model_out.metadata["framework"] = "sklearn"
    model_out.metadata["training_date"] = "2025-01-15"
    model_out.metadata["git_commit"] = "abc123"
    model_out.metadata["dataset_version"] = "v2.3"
```

---

## Hands-On Lab

### Exercise 1: Query MLMD Programmatically

```python
"""query_metadata.py -- query KFP metadata store."""
from kfp.client import Client


def explore_run_artifacts(kfp_host: str, run_id: str) -> None:
    """List all artifacts produced by a specific pipeline run."""
    client = Client(host=kfp_host)

    run = client.get_run(run_id=run_id)
    print(f"Run: {run.display_name} ({run.state})")
    print(f"Created: {run.created_at}")
    print(f"Pipeline: {run.pipeline_spec.pipeline_name}")

    # The KFP v2 API exposes artifacts through the run details
    # In practice, you query the MLMD gRPC API or use the KFP REST API
    print("\nTo explore artifacts in the UI:")
    print(f"  {kfp_host}/#/runs/details/{run_id}")


def list_artifacts_via_rest(kfp_host: str) -> None:
    """List recent artifacts using the KFP REST API."""
    import requests

    resp = requests.get(
        f"{kfp_host}/apis/v2beta1/artifacts",
        params={"page_size": 20, "order_by": "create_time desc"},
    )
    resp.raise_for_status()
    artifacts = resp.json().get("artifacts", [])

    print(f"\nRecent artifacts ({len(artifacts)}):")
    for a in artifacts:
        print(f"  ID: {a['artifact_id']}")
        print(f"  Type: {a.get('artifact_type', 'unknown')}")
        print(f"  URI: {a.get('uri', 'N/A')}")
        print(f"  Custom props: {a.get('custom_properties', {})}")
        print()


if __name__ == "__main__":
    explore_run_artifacts("http://localhost:8080", run_id="<YOUR_RUN_ID>")
    list_artifacts_via_rest("http://localhost:8080")
```

### Exercise 2: Lineage-Aware Pipeline

Build a pipeline that explicitly records provenance metadata at every step:

```python
from kfp import dsl
from kfp.dsl import Dataset, Model, Metrics, Input, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1", "scikit-learn>=1.3"],
)
def load_with_lineage(
    dataset_name: str,
    dataset_version: str,
    output_data: Output[Dataset],
    lineage_info: Output[Metrics],
) -> None:
    """Load data and record detailed provenance."""
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    import hashlib
    from datetime import datetime

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df.to_csv(output_data.path, index=False)

    # Record lineage metadata on the artifact
    output_data.metadata["dataset_name"] = dataset_name
    output_data.metadata["dataset_version"] = dataset_version
    output_data.metadata["num_rows"] = len(df)
    output_data.metadata["num_columns"] = len(df.columns)
    output_data.metadata["loaded_at"] = datetime.utcnow().isoformat()

    # Compute a content hash for data versioning
    content_hash = hashlib.sha256(
        df.to_csv(index=False).encode()
    ).hexdigest()[:16]
    output_data.metadata["content_hash"] = content_hash

    lineage_info.log_metric("dataset_name", dataset_name)
    lineage_info.log_metric("dataset_version", dataset_version)
    lineage_info.log_metric("content_hash", content_hash)
    lineage_info.log_metric("num_rows", len(df))

    print(f"Loaded {dataset_name} v{dataset_version} (hash={content_hash})")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.1", "scikit-learn>=1.3", "joblib>=1.3",
    ],
)
def train_with_lineage(
    train_data: Input[Dataset],
    model_out: Output[Model],
    training_log: Output[Metrics],
    algorithm: str = "random_forest",
    git_commit: str = "unknown",
) -> None:
    """Train a model and record full provenance metadata."""
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from datetime import datetime

    df = pd.read_csv(train_data.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipe.fit(X, y)
    accuracy = pipe.score(X, y)

    joblib.dump(pipe, model_out.path)

    # Record lineage on model artifact
    model_out.metadata["algorithm"] = algorithm
    model_out.metadata["framework"] = "sklearn"
    model_out.metadata["train_accuracy"] = round(accuracy, 4)
    model_out.metadata["trained_at"] = datetime.utcnow().isoformat()
    model_out.metadata["git_commit"] = git_commit
    model_out.metadata["input_data_hash"] = train_data.metadata.get(
        "content_hash", "unknown"
    )

    training_log.log_metric("algorithm", algorithm)
    training_log.log_metric("train_accuracy", round(accuracy, 4))
    training_log.log_metric("git_commit", git_commit)

    print(f"Model trained: accuracy={accuracy:.4f}, git={git_commit}")
```

### Exercise 3: MLflow Integration for Experiment Tracking

```python
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "mlflow>=2.9", "pandas>=2.1",
        "scikit-learn>=1.3", "joblib>=1.3", "boto3>=1.34",
    ],
)
def train_with_mlflow(
    train_data: dsl.Input[dsl.Dataset],
    model_out: dsl.Output[dsl.Model],
    algorithm: str = "random_forest",
    n_estimators: int = 100,
    mlflow_tracking_uri: str = "http://mlflow:5000",
    mlflow_experiment: str = "kfp-pipeline",
) -> float:
    """Train a model with full MLflow experiment tracking."""
    import pandas as pd
    import joblib
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    df = pd.read_csv(train_data.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, random_state=42,
        )),
    ])

    with mlflow.start_run(run_name=f"kfp-{algorithm}-{n_estimators}") as run:
        # Log parameters
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("n_estimators", n_estimators)

        # Train
        pipe.fit(X, y)
        train_acc = pipe.score(X, y)

        # Cross-validation
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")

        # Log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())

        # Log model
        mlflow.sklearn.log_model(
            pipe, artifact_path="model",
            registered_model_name=f"kfp-{algorithm}",
        )

        print(f"MLflow run: {run.info.run_id}")
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    joblib.dump(pipe, model_out.path)
    model_out.metadata["mlflow_run_id"] = run.info.run_id

    return train_acc
```

---

## Metadata and Lineage Patterns

| Pattern | Description |
|---|---|
| **Content hashing** | Hash dataset contents for data versioning |
| **Git commit tagging** | Attach the code commit SHA to every model artifact |
| **Timestamp recording** | Record when each step executed |
| **Upstream reference** | Store the input artifact ID/hash on the output artifact |
| **MLflow cross-reference** | Store the MLflow run ID in the KFP artifact metadata |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Not recording metadata on artifacts | Cannot trace model to data | Use `.metadata["key"] = value` on every output |
| Relying only on KFP UI for lineage | Cannot query programmatically | Use the MLMD gRPC API or KFP REST API |
| No data versioning | Cannot reproduce a training run | Hash data content or use DVC/lakehouse versioning |
| MLflow URI hardcoded | Works locally, fails on cluster | Use environment variables or pipeline parameters |

---

## Self-Check Questions

1. What four entities does MLMD track?
2. How would you trace a deployed model back to the exact dataset it was trained on?
3. What is the purpose of content hashing for data lineage?
4. How do KFP metadata and MLflow complement each other?
5. What query would you run to find all models affected by a known data quality issue?

---

## You Know You Have Completed This Module When...

- [ ] Can explain the MLMD data model (Artifact, Execution, Context, Event)
- [ ] Added custom metadata to component outputs
- [ ] Queried run artifacts programmatically
- [ ] Integrated MLflow for experiment tracking
- [ ] Built a lineage-aware pipeline with provenance at every step

---

**Next: [Module 09 -- Metadata and Data Lineage -->](../09-metadata-and-lineage/)**
