# Module 03: Pipeline Definition -- DAGs, Parameters, and Conditions

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 02 completed, familiar with KFP components |

---

## Learning Objectives

By the end of this module, you will be able to:

- Define complex pipeline DAGs with branching and merging
- Use pipeline parameters to make pipelines configurable
- Apply conditional execution with `dsl.If` / `dsl.Else`
- Use `dsl.ParallelFor` for fan-out/fan-in patterns
- Set resource requests, timeouts, and retry policies on tasks
- Compile and inspect the pipeline IR YAML

---

## Concepts

### Pipeline as a DAG

A KFP pipeline is a **Directed Acyclic Graph (DAG)**. Each node is a component task; edges represent data dependencies. KFP infers the execution order from how you wire outputs to inputs -- you never specify the order explicitly.

```
          +------------+
          | Load Data  |
          +-----+------+
                |
        +-------+-------+
        |               |
  +-----v-----+   +-----v------+
  | Validate  |   | Profile    |
  +-----+-----+   +-----+------+
        |               |
        +-------+-------+
                |
          +-----v------+
          | Transform  |
          +-----+------+
                |
        +-------+-------+
        |               |
  +-----v-----+   +-----v------+
  | Train RF  |   | Train XGB  |
  +-----+-----+   +-----+------+
        |               |
        +-------+-------+
                |
          +-----v------+
          | Compare &  |
          | Select     |
          +-----+------+
                |
          +-----v------+
          | Deploy     |
          +------------+
```

### Pipeline Parameters

Parameters make pipelines reusable. Declare them as function arguments with type hints and defaults:

```python
from kfp import dsl


@dsl.pipeline(
    name="configurable-training-pipeline",
    description="A pipeline with runtime-configurable parameters.",
)
def training_pipeline(
    # String parameters
    dataset_uri: str = "gs://my-bucket/data.csv",
    model_type: str = "random_forest",
    # Numeric parameters
    n_estimators: int = 100,
    learning_rate: float = 0.01,
    test_split: float = 0.2,
    # Boolean parameters
    enable_profiling: bool = False,
) -> None:
    """All parameters can be overridden from the KFP UI or CLI."""
    load_task = load_data(uri=dataset_uri, test_split=test_split)
    train_task = train_model(
        data=load_task.outputs["train_data"],
        model_type=model_type,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
    )
```

When submitting via the Python client:

```python
from kfp.client import Client

client = Client(host="http://localhost:8080")
client.create_run_from_pipeline_package(
    pipeline_file="pipeline.yaml",
    arguments={
        "dataset_uri": "gs://prod-bucket/jan_2025.csv",
        "model_type": "xgboost",
        "n_estimators": 500,
        "learning_rate": 0.005,
    },
)
```

### Conditional Execution

Use `dsl.If`, `dsl.Elif`, and `dsl.Else` to branch the pipeline at runtime:

```python
@dsl.pipeline(name="conditional-pipeline")
def conditional_pipeline(accuracy_threshold: float = 0.90) -> None:
    load_task = data_loader(dataset_name="breast_cancer")
    train_task = trainer(data=load_task.outputs["train_data"])
    eval_task = evaluator(
        data=load_task.outputs["test_data"],
        model=train_task.outputs["model"],
    )

    with dsl.If(eval_task.output == "deploy"):
        deploy_task = deploy_model(model=train_task.outputs["model"])
        notify_task = send_notification(message="Model deployed successfully")

    with dsl.Else():
        retrain_task = retrain_with_tuning(
            data=load_task.outputs["train_data"],
        )
        alert_task = send_notification(message="Model below threshold; retraining")
```

### Parallel Execution with ParallelFor

Fan out over a list and collect results:

```python
@dsl.pipeline(name="parallel-training-pipeline")
def parallel_training(
    dataset_name: str = "breast_cancer",
    model_types: list = ["random_forest", "xgboost", "logistic_regression"],
) -> None:
    load_task = data_loader(dataset_name=dataset_name)

    with dsl.ParallelFor(model_types) as model_type:
        train_task = trainer(
            data=load_task.outputs["train_data"],
            model_type=model_type,
        )
        eval_task = evaluator(
            data=load_task.outputs["test_data"],
            model=train_task.outputs["model"],
        )
```

---

## Hands-On Lab

### Exercise 1: Build a Multi-Branch Pipeline

Create `multi_branch_pipeline.py`:

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
) -> None:
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1", "scikit-learn>=1.3", "joblib>=1.3"],
)
def train_sklearn(
    train_data: Input[Dataset],
    model_out: Output[Model],
    metrics_out: Output[Metrics],
    algorithm: str = "random_forest",
    n_estimators: int = 100,
) -> float:
    import pandas as pd, joblib
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    df = pd.read_csv(train_data.path)
    X, y = df.drop(columns=["target"]), df["target"]

    if algorithm == "random_forest":
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)

    clf.fit(X, y)
    acc = clf.score(X, y)
    joblib.dump(clf, model_out.path)

    metrics_out.log_metric("train_accuracy", round(acc, 4))
    metrics_out.log_metric("algorithm", algorithm)
    return acc


@dsl.component(base_image="python:3.11-slim")
def pick_best(acc_rf: float, acc_gb: float) -> str:
    if acc_rf >= acc_gb:
        print(f"Random Forest wins: {acc_rf:.4f} vs {acc_gb:.4f}")
        return "random_forest"
    else:
        print(f"Gradient Boosting wins: {acc_gb:.4f} vs {acc_rf:.4f}")
        return "gradient_boosting"


@dsl.pipeline(name="multi-branch-pipeline")
def multi_branch_pipeline(
    dataset_name: str = "breast_cancer",
    test_size: float = 0.2,
    n_estimators: int = 100,
) -> None:
    load_task = load_and_split(dataset_name=dataset_name, test_size=test_size)

    # Branch 1: Random Forest
    rf_task = train_sklearn(
        train_data=load_task.outputs["train_data"],
        algorithm="random_forest",
        n_estimators=n_estimators,
    )
    rf_task.set_display_name("Train Random Forest")

    # Branch 2: Gradient Boosting
    gb_task = train_sklearn(
        train_data=load_task.outputs["train_data"],
        algorithm="gradient_boosting",
        n_estimators=n_estimators,
    )
    gb_task.set_display_name("Train Gradient Boosting")

    # Merge: pick best
    best_task = pick_best(acc_rf=rf_task.output, acc_gb=gb_task.output)
    best_task.set_display_name("Select Best Model")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=multi_branch_pipeline,
        package_path="multi_branch_pipeline.yaml",
    )
    print("Compiled multi_branch_pipeline.yaml")
```

### Exercise 2: Add Resource Constraints and Retries

```python
@dsl.pipeline(name="resource-aware-pipeline")
def resource_aware_pipeline(dataset_name: str = "breast_cancer") -> None:
    load_task = load_and_split(dataset_name=dataset_name, test_size=0.2)

    train_task = train_sklearn(
        train_data=load_task.outputs["train_data"],
        algorithm="random_forest",
    )

    # Resource requests and limits
    train_task.set_cpu_limit("2")
    train_task.set_memory_limit("4Gi")
    train_task.set_cpu_request("1")
    train_task.set_memory_request("2Gi")

    # Retry policy for transient failures
    train_task.set_retry(
        num_retries=3,
        backoff_duration="60s",
        backoff_factor=2.0,
    )

    # Timeout: 1 hour max
    train_task.set_timeout(seconds=3600)
```

### Exercise 3: Exit Handler for Cleanup

```python
@dsl.component(base_image="python:3.11-slim")
def cleanup(status: str) -> None:
    """Runs after the pipeline finishes, regardless of outcome."""
    print(f"Pipeline finished with status: {status}")
    print("Cleaning up temporary resources...")


@dsl.pipeline(name="pipeline-with-cleanup")
def pipeline_with_cleanup(dataset_name: str = "breast_cancer") -> None:
    exit_task = cleanup(status="completed")

    with dsl.ExitHandler(exit_task=exit_task):
        load_task = load_and_split(dataset_name=dataset_name, test_size=0.2)
        train_task = train_sklearn(
            train_data=load_task.outputs["train_data"],
        )
```

---

## Pipeline Design Patterns

| Pattern | Use case | KFP construct |
|---|---|---|
| **Sequential** | Steps that depend on each other | Wire output to input |
| **Parallel branches** | Independent model training | Multiple tasks from the same input |
| **Fan-out / fan-in** | Hyperparameter sweep | `dsl.ParallelFor` + `dsl.Collected` |
| **Conditional** | Deploy only if metrics pass | `dsl.If` / `dsl.Else` |
| **Exit handler** | Cleanup on success or failure | `dsl.ExitHandler` |
| **Nested pipeline** | Reusable sub-workflows | Call a `@dsl.pipeline` inside another |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Circular dependency | Compilation error | Ensure the DAG has no cycles |
| Using `dsl.If` with an artifact | Type error | Conditions must compare primitive return values |
| Not setting resource limits | Pod evicted by K8s | Always set `set_memory_limit()` for training |
| Python `if` instead of `dsl.If` | Branch always taken at compile time | Use `dsl.If` for runtime branching |

---

## Self-Check Questions

1. How does KFP determine the execution order of tasks?
2. What is the difference between `dsl.If` and a Python `if` in a pipeline definition?
3. How would you train 10 different hyperparameter sets in parallel?
4. What happens if a task inside a `dsl.ExitHandler` fails?
5. How do you inspect the compiled IR YAML to debug a pipeline?

---

## You Know You Have Completed This Module When...

- [ ] Built a multi-branch pipeline with parallel training
- [ ] Used `dsl.If` for conditional deployment
- [ ] Set resource limits, retries, and timeouts on tasks
- [ ] Compiled a pipeline and inspected the IR YAML
- [ ] Can explain DAG execution order from a pipeline definition

---

**Next: [Module 04 -- Data Processing Pipeline Steps -->](../04-data-processing-steps/)**
