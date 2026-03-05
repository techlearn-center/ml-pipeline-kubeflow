# Module 10: Production Patterns -- CI/CD for Pipelines and Monitoring

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Advanced |
| **Prerequisites** | Module 09 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Implement CI/CD for ML pipelines using GitHub Actions
- Version pipelines with semantic versioning and git tags
- Monitor model performance in production (data drift, prediction drift)
- Set up alerting for pipeline failures and model degradation
- Apply infrastructure best practices for production Kubeflow

---

## Concepts

### CI/CD for ML Pipelines

CI/CD for ML is different from traditional software CI/CD. You are not just testing code -- you are testing data, model quality, and pipeline behaviour.

```
  +----------+    +-----------+    +----------+    +----------+    +----------+
  | Git Push |--->| Lint +    |--->| Compile  |--->| Test     |--->| Deploy   |
  |          |    | Unit Test |    | Pipeline |    | Pipeline |    | Pipeline |
  +----------+    +-----------+    +----------+    +----------+    +----------+
                       |                |               |               |
                  Code quality     IR YAML valid    Run on test     Upload to
                  checks           and correct      dataset         production KFP
```

| CI/CD Stage | What it does | Tools |
|---|---|---|
| **Lint** | Check code style, type hints | ruff, mypy |
| **Unit test** | Test component logic without KFP | pytest |
| **Compile** | Compile pipeline to IR YAML | `kfp.compiler` |
| **Integration test** | Run pipeline on test data | KFP on a test cluster |
| **Deploy** | Upload pipeline to production KFP | `kfp.Client.upload_pipeline()` |

### Model Monitoring in Production

Once a model is deployed, you must watch for:

| Issue | Description | Detection |
|---|---|---|
| **Data drift** | Input feature distributions shift | Statistical tests (KS, PSI) |
| **Prediction drift** | Model output distribution shifts | Monitor prediction histograms |
| **Performance degradation** | Accuracy drops on new data | Compare with ground truth labels |
| **Latency increase** | Model serving slows down | p50/p95/p99 latency metrics |
| **Error rate spike** | Serving errors increase | HTTP 5xx rate monitoring |

---

## Hands-On Lab

### Exercise 1: GitHub Actions CI/CD Pipeline

Create `.github/workflows/ml-pipeline-ci.yml`:

```yaml
name: ML Pipeline CI/CD

on:
  push:
    branches: [main]
    paths:
      - "src/**"
      - "requirements.txt"
  pull_request:
    branches: [main]
    paths:
      - "src/**"

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install ruff pytest

      - name: Lint
        run: ruff check src/

      - name: Unit tests
        run: pytest tests/ -v --tb=short

  compile-pipeline:
    runs-on: ubuntu-latest
    needs: lint-and-test
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Compile pipeline
        run: python -m src.pipelines.training_pipeline

      - name: Validate IR YAML
        run: |
          python -c "
          import yaml
          with open('pipeline.yaml') as f:
              ir = yaml.safe_load(f)
          assert 'pipelineInfo' in ir, 'Missing pipelineInfo'
          assert 'components' in ir, 'Missing components'
          print('Pipeline IR is valid')
          print(f'  Components: {len(ir[\"components\"])}')
          "

      - name: Upload pipeline artifact
        uses: actions/upload-artifact@v4
        with:
          name: pipeline-yaml
          path: pipeline.yaml

  deploy-pipeline:
    runs-on: ubuntu-latest
    needs: compile-pipeline
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Download pipeline artifact
        uses: actions/download-artifact@v4
        with:
          name: pipeline-yaml

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install KFP client
        run: pip install "kfp>=2.0"

      - name: Deploy to KFP
        env:
          KFP_ENDPOINT: ${{ secrets.KFP_ENDPOINT }}
        run: |
          python -c "
          from kfp.client import Client
          import os

          client = Client(host=os.environ['KFP_ENDPOINT'])
          pipeline = client.upload_pipeline(
              pipeline_package_path='pipeline.yaml',
              pipeline_name='training-pipeline',
              description='Automated deploy from CI/CD',
          )
          print(f'Pipeline uploaded: {pipeline.pipeline_id}')
          "
```

### Exercise 2: Unit Testing Components Locally

Create `tests/test_components.py`:

```python
"""test_components.py -- unit tests for pipeline component logic."""
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@pytest.fixture
def breast_cancer_data():
    """Load breast cancer dataset split into train/test."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"],
    )
    return train_df, test_df


def test_data_loading():
    """Test that data loading produces expected shapes."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    assert len(df) == 569
    assert "target" in df.columns
    assert df["target"].nunique() == 2


def test_train_test_split_proportions(breast_cancer_data):
    """Test that the split maintains approximate proportions."""
    train_df, test_df = breast_cancer_data
    total = len(train_df) + len(test_df)

    assert abs(len(test_df) / total - 0.2) < 0.02
    assert len(train_df) > len(test_df)


def test_model_training(breast_cancer_data):
    """Test that model trains and achieves reasonable accuracy."""
    train_df, test_df = breast_cancer_data
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    train_acc = pipe.score(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)

    assert train_acc > 0.90, f"Train accuracy too low: {train_acc}"
    assert test_acc > 0.85, f"Test accuracy too low: {test_acc}"


def test_evaluation_metrics(breast_cancer_data):
    """Test that evaluation produces valid metrics."""
    from sklearn.metrics import accuracy_score, f1_score

    train_df, test_df = breast_cancer_data
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert acc > 0.85
    assert f1 > 0.80


def test_deploy_decision():
    """Test the deploy/no_deploy decision logic."""
    def make_decision(accuracy, f1, acc_threshold=0.85, f1_threshold=0.80):
        if accuracy >= acc_threshold and f1 >= f1_threshold:
            return "deploy"
        return "no_deploy"

    assert make_decision(0.95, 0.90) == "deploy"
    assert make_decision(0.80, 0.90) == "no_deploy"
    assert make_decision(0.95, 0.70) == "no_deploy"
    assert make_decision(0.85, 0.80) == "deploy"
    assert make_decision(0.84, 0.80) == "no_deploy"
```

### Exercise 3: Data Drift Detection Component

```python
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1", "numpy>=1.25", "scipy>=1.11"],
)
def detect_data_drift(
    reference_data: Input[Dataset],
    production_data: Input[Dataset],
    drift_report: Output[Metrics],
    drift_threshold: float = 0.05,
) -> str:
    """Detect data drift using the Kolmogorov-Smirnov test.

    Returns 'drift_detected' or 'no_drift'.
    """
    import pandas as pd
    import numpy as np
    from scipy import stats

    ref_df = pd.read_csv(reference_data.path)
    prod_df = pd.read_csv(production_data.path)

    numeric_cols = ref_df.select_dtypes(include=[np.number]).columns
    drifted_features = []

    for col in numeric_cols:
        if col == "target":
            continue

        ref_values = ref_df[col].dropna()
        prod_values = prod_df[col].dropna()

        ks_stat, p_value = stats.ks_2samp(ref_values, prod_values)

        drift_report.log_metric(f"ks_stat_{col}", round(ks_stat, 4))
        drift_report.log_metric(f"p_value_{col}", round(p_value, 4))

        if p_value < drift_threshold:
            drifted_features.append(col)
            print(f"DRIFT in '{col}': KS={ks_stat:.4f}, p={p_value:.4f}")

    drift_report.log_metric("num_features_checked", len(numeric_cols) - 1)
    drift_report.log_metric("num_features_drifted", len(drifted_features))
    drift_report.log_metric("drift_pct", round(
        len(drifted_features) / max(len(numeric_cols) - 1, 1), 4
    ))

    if drifted_features:
        print(f"\nDrift detected in {len(drifted_features)} features")
        return "drift_detected"
    else:
        print("No significant drift detected")
        return "no_drift"
```

### Exercise 4: Monitoring Pipeline

```python
from kfp import dsl, compiler


@dsl.pipeline(
    name="monitoring-pipeline",
    description="Detect data drift and trigger retraining if needed.",
)
def monitoring_pipeline(
    reference_dataset: str = "gs://my-bucket/reference_data.csv",
    production_dataset: str = "gs://my-bucket/latest_production_data.csv",
    drift_threshold: float = 0.05,
    retrain_model_type: str = "random_forest",
) -> None:
    # Step 1: Load reference and production data
    ref_task = load_data(source=reference_dataset)
    prod_task = load_data(source=production_dataset)

    # Step 2: Detect drift
    drift_task = detect_data_drift(
        reference_data=ref_task.outputs["output_data"],
        production_data=prod_task.outputs["output_data"],
        drift_threshold=drift_threshold,
    )
    drift_task.set_display_name("Detect Data Drift")

    # Step 3: Conditionally retrain
    with dsl.If(drift_task.output == "drift_detected"):
        notify_task = notify(
            message="Data drift detected -- triggering retraining",
            channel="slack",
        )
        retrain_task = train_candidate(
            train_data=prod_task.outputs["output_data"],
            algorithm=retrain_model_type,
        )
        retrain_task.set_display_name("Retrain on New Data")

    with dsl.Else():
        no_drift_notify = notify(
            message="No drift detected -- model is healthy",
            channel="slack",
        )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=monitoring_pipeline,
        package_path="monitoring_pipeline.yaml",
    )
```

### Exercise 5: Pipeline Versioning Script

```python
"""version_pipeline.py -- version and upload a pipeline to KFP."""
import argparse
import subprocess
from kfp import compiler
from kfp.client import Client
from src.pipelines.training_pipeline import training_pipeline


def get_git_info() -> dict:
    """Get current git commit and tag info."""
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]
    ).decode().strip()

    try:
        tag = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"]
        ).decode().strip()
    except subprocess.CalledProcessError:
        tag = "v0.0.0"

    return {"commit": commit, "tag": tag}


def main():
    parser = argparse.ArgumentParser(description="Version and deploy pipeline")
    parser.add_argument("--kfp-host", default="http://localhost:8080")
    parser.add_argument("--version", help="Pipeline version (default: git tag)")
    args = parser.parse_args()

    git_info = get_git_info()
    version = args.version or git_info["tag"]

    # Compile
    pipeline_file = f"training_pipeline_{version}.yaml"
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=pipeline_file,
    )
    print(f"Compiled: {pipeline_file}")

    # Upload
    client = Client(host=args.kfp_host)
    pipeline = client.upload_pipeline(
        pipeline_package_path=pipeline_file,
        pipeline_name=f"training-pipeline",
        description=f"Version {version} (commit {git_info['commit']})",
    )
    print(f"Uploaded pipeline: {pipeline.pipeline_id}")
    print(f"Version: {version}, Commit: {git_info['commit']}")


if __name__ == "__main__":
    main()
```

---

## Production Checklist

| Category | Item | Status |
|---|---|---|
| **CI/CD** | Pipeline compiles on every PR | |
| **CI/CD** | Component unit tests pass | |
| **CI/CD** | Pipeline auto-deploys on merge to main | |
| **Monitoring** | Data drift detection scheduled | |
| **Monitoring** | Model performance tracked | |
| **Monitoring** | Alerting for pipeline failures | |
| **Infra** | Resource limits on all components | |
| **Infra** | Secrets managed (not hardcoded) | |
| **Infra** | Pipeline versioned with git tags | |
| **Reliability** | Retry policies on flaky steps | |
| **Reliability** | Exit handlers for cleanup | |
| **Reliability** | Caching for expensive steps | |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| No CI/CD | Broken pipelines deployed manually | Set up GitHub Actions or similar |
| No monitoring | Model degrades silently | Schedule drift detection pipeline |
| Secrets in code | Credentials exposed in git | Use K8s secrets or vault |
| No versioning | Cannot roll back | Tag pipelines with semantic versions |
| Manual testing only | Regressions slip through | Write pytest tests for component logic |

---

## Self-Check Questions

1. What are the stages of a CI/CD pipeline for ML?
2. How would you detect data drift between training data and production data?
3. Why should pipeline deployment be automated rather than manual?
4. What metrics would you monitor for a deployed classification model?
5. How do you roll back a pipeline to a previous version?

---

## You Know You Have Completed This Module When...

- [ ] Created a GitHub Actions workflow for pipeline CI/CD
- [ ] Wrote unit tests for component logic
- [ ] Built a data drift detection component
- [ ] Implemented pipeline versioning with git tags
- [ ] Can describe a complete production monitoring strategy

---

**Next: [Capstone Project -->](../../capstone/)**
