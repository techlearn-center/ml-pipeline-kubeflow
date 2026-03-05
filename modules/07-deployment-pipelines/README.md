# Module 07: Pipeline Scheduling and Triggers

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 06 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Schedule recurring pipeline runs with cron expressions
- Trigger pipelines on events (new data, model drift, API call)
- Manage experiments and runs programmatically with the KFP Python client
- Implement run-level parameter overrides for scheduled jobs
- Monitor scheduled run health and set up failure alerts

---

## Concepts

### Why Scheduling Matters

A model trained once is a demo. A model retrained on schedule is a product. In production, pipelines must run automatically:

| Trigger type | Example | KFP mechanism |
|---|---|---|
| **Time-based (cron)** | Retrain every Monday at 2 AM | `client.create_recurring_run()` |
| **Event-based** | New data lands in S3 | External trigger via KFP REST API |
| **On-demand** | Data scientist clicks "Run" | KFP UI or `client.create_run_from_pipeline_package()` |
| **CI/CD** | Pipeline code merged to main | GitHub Actions calls KFP API |

### KFP Python Client

The `kfp.Client` is your programmatic interface to the KFP API server:

```python
from kfp.client import Client

client = Client(host="http://localhost:8080")

# List experiments
experiments = client.list_experiments()
for exp in experiments.experiments:
    print(f"  {exp.experiment_id}: {exp.display_name}")

# List pipelines
pipelines = client.list_pipelines()
for p in pipelines.pipelines:
    print(f"  {p.pipeline_id}: {p.display_name}")
```

---

## Hands-On Lab

### Exercise 1: Submit a Run Programmatically

```python
"""submit_run.py -- submit a single pipeline run via the KFP client."""
from kfp.client import Client
from kfp import compiler
from src.pipelines.training_pipeline import training_pipeline

# Step 1: Compile
compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path="training_pipeline.yaml",
)

# Step 2: Connect to KFP
client = Client(host="http://localhost:8080")

# Step 3: Create or get experiment
experiment = client.create_experiment(
    name="scheduled-training",
    description="Experiment for automated retraining runs",
)
print(f"Experiment: {experiment.experiment_id}")

# Step 4: Submit run with custom parameters
run = client.create_run_from_pipeline_package(
    pipeline_file="training_pipeline.yaml",
    arguments={
        "dataset_name": "breast_cancer",
        "model_type": "random_forest",
        "n_estimators": 200,
        "accuracy_threshold": 0.90,
    },
    run_name="manual-run-2025-01-15",
    experiment_name="scheduled-training",
)

print(f"Run submitted: {run.run_id}")
print(f"Run URL: http://localhost:8080/#/runs/details/{run.run_id}")
```

### Exercise 2: Create a Recurring (Cron) Schedule

```python
"""schedule_pipeline.py -- create a recurring pipeline run."""
from kfp.client import Client
from kfp import compiler
from src.pipelines.training_pipeline import training_pipeline

# Compile
compiler.Compiler().compile(
    pipeline_func=training_pipeline,
    package_path="training_pipeline.yaml",
)

client = Client(host="http://localhost:8080")

# Create experiment
experiment = client.create_experiment(name="weekly-retraining")

# Upload pipeline
pipeline = client.upload_pipeline(
    pipeline_package_path="training_pipeline.yaml",
    pipeline_name="training-pipeline-v1",
    description="Weekly retraining pipeline",
)

# Schedule: every Monday at 2:00 AM UTC
recurring_run = client.create_recurring_run(
    experiment_id=experiment.experiment_id,
    job_name="weekly-retrain-breast-cancer",
    pipeline_id=pipeline.pipeline_id,
    cron_expression="0 2 * * 1",  # minute hour day-of-month month day-of-week
    max_concurrency=1,
    enabled=True,
    parameters={
        "dataset_name": "breast_cancer",
        "model_type": "random_forest",
        "n_estimators": 200,
        "accuracy_threshold": 0.90,
    },
)

print(f"Recurring run created: {recurring_run.job_id}")
print(f"Schedule: every Monday at 02:00 UTC")
```

### Exercise 3: Event-Driven Trigger via REST API

When new data arrives, an external service can trigger the pipeline:

```python
"""trigger_pipeline.py -- trigger a pipeline run via REST API."""
import requests
import json


def trigger_pipeline_run(
    kfp_host: str = "http://localhost:8080",
    pipeline_name: str = "training-pipeline-v1",
    parameters: dict = None,
    run_name: str = "event-triggered-run",
) -> str:
    """Trigger a KFP pipeline run via the REST API."""

    # Step 1: Find pipeline ID by name
    resp = requests.get(f"{kfp_host}/apis/v2beta1/pipelines")
    resp.raise_for_status()
    pipelines = resp.json().get("pipelines", [])

    pipeline_id = None
    for p in pipelines:
        if p["display_name"] == pipeline_name:
            pipeline_id = p["pipeline_id"]
            break

    if not pipeline_id:
        raise ValueError(f"Pipeline '{pipeline_name}' not found")

    # Step 2: Get the default pipeline version
    resp = requests.get(
        f"{kfp_host}/apis/v2beta1/pipelines/{pipeline_id}/versions"
    )
    resp.raise_for_status()
    versions = resp.json().get("pipeline_versions", [])
    version_id = versions[0]["pipeline_version_id"]

    # Step 3: Create a run
    run_body = {
        "display_name": run_name,
        "pipeline_version_reference": {
            "pipeline_id": pipeline_id,
            "pipeline_version_id": version_id,
        },
        "runtime_config": {
            "parameters": parameters or {},
        },
    }

    resp = requests.post(
        f"{kfp_host}/apis/v2beta1/runs",
        json=run_body,
    )
    resp.raise_for_status()
    run_id = resp.json()["run_id"]
    print(f"Triggered run: {run_id}")
    return run_id


# Example: data landing event triggers retraining
if __name__ == "__main__":
    trigger_pipeline_run(
        parameters={
            "dataset_name": "gs://my-bucket/new_data_2025_01.csv",
            "model_type": "xgboost",
        },
        run_name="data-landing-trigger-jan-2025",
    )
```

### Exercise 4: Monitor Run Status

```python
"""monitor_runs.py -- check on pipeline run status."""
from kfp.client import Client
import time


def wait_for_run(client: Client, run_id: str, timeout: int = 3600) -> str:
    """Poll until a run completes or times out."""
    start = time.time()

    while time.time() - start < timeout:
        run = client.get_run(run_id=run_id)
        status = run.state

        print(f"  Run {run_id}: {status}")

        if status in ("SUCCEEDED", "FAILED", "SKIPPED", "ERROR"):
            return status

        time.sleep(30)

    raise TimeoutError(f"Run {run_id} did not finish in {timeout}s")


def list_recent_runs(client: Client, experiment_name: str, limit: int = 10):
    """List recent runs for an experiment."""
    experiment = client.get_experiment(experiment_name=experiment_name)
    runs = client.list_runs(
        experiment_id=experiment.experiment_id,
        page_size=limit,
        sort_by="created_at desc",
    )
    print(f"\nRecent runs for '{experiment_name}':")
    for run in runs.runs or []:
        print(f"  {run.display_name}: {run.state} ({run.created_at})")


if __name__ == "__main__":
    client = Client(host="http://localhost:8080")
    list_recent_runs(client, "weekly-retraining", limit=5)
```

---

## Cron Expression Reference

| Expression | Meaning |
|---|---|
| `0 2 * * 1` | Every Monday at 02:00 UTC |
| `0 0 * * *` | Every day at midnight |
| `0 */6 * * *` | Every 6 hours |
| `0 9 1 * *` | First day of every month at 09:00 |
| `30 14 * * 1-5` | Weekdays at 14:30 |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Cron schedule in local time | Runs at unexpected hours | KFP uses UTC; convert your timezone |
| `max_concurrency` not set | Overlapping runs compete for resources | Set `max_concurrency=1` |
| No failure alerting | Silent failures go unnoticed | Add a monitoring step or webhook |
| Hardcoded `kfp_host` | Works locally, fails in CI | Use environment variables |

---

## Self-Check Questions

1. How do you create a recurring pipeline run that executes every Sunday at midnight?
2. What is the difference between `create_run_from_pipeline_package` and `create_recurring_run`?
3. How would an external system (e.g., Airflow, cloud function) trigger a KFP run?
4. What does `max_concurrency=1` ensure?
5. How would you monitor for failed recurring runs?

---

## You Know You Have Completed This Module When...

- [ ] Submitted a pipeline run programmatically
- [ ] Created a recurring schedule with a cron expression
- [ ] Triggered a pipeline via the REST API
- [ ] Monitored run status and listed recent runs
- [ ] Can explain the difference between cron, event, and on-demand triggers

---

**Next: [Module 08 -- Pipeline Scheduling -->](../08-pipeline-scheduling/)**
