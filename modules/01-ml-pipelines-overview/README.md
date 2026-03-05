# Module 01: ML Pipelines Overview -- Kubeflow Fundamentals

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner |
| **Prerequisites** | Python 3.10+, Docker installed, basic terminal knowledge |

---

## Learning Objectives

By the end of this module, you will be able to:

- Explain the purpose and architecture of Kubeflow Pipelines (KFP)
- Describe how KFP components, pipelines, and runs relate to each other
- Install the KFP v2 SDK and verify your local environment
- Compile and inspect a minimal "hello world" pipeline
- Navigate the Kubeflow Pipelines UI

---

## Concepts

### What Are ML Pipelines?

An **ML pipeline** is a sequence of automated, reproducible steps that take raw data and produce a trained, validated, deployable model. In production, teams never train models by running notebooks manually. Instead they encode every step -- data ingestion, feature engineering, training, evaluation, deployment -- into a pipeline that can be versioned, tested, scheduled, and audited.

**Why pipelines matter in production:**

| Manual workflow | Pipeline-based workflow |
|---|---|
| Notebooks run by hand on a laptop | Automated, serverless execution on Kubernetes |
| "It works on my machine" | Containerised, reproducible environments |
| No audit trail | Full metadata and lineage for every run |
| Retraining is a weekend task | Retraining is triggered by a cron job or data event |

### Kubeflow Pipelines Architecture

Kubeflow Pipelines runs on Kubernetes and consists of several key services:

```
                    +---------------------+
                    |   KFP UI (React)    |
                    +----------+----------+
                               |
                    +----------v----------+
                    |   KFP API Server    |
                    +----------+----------+
                               |
          +--------------------+--------------------+
          |                    |                    |
+---------v--------+ +---------v--------+ +--------v---------+
|  Argo Workflows  | |  ML Metadata     | |  Artifact Store  |
|  (orchestrator)  | |  (MLMD / MySQL)  | |  (MinIO / GCS)   |
+------------------+ +------------------+ +------------------+
          |
+---------v--------------------------------------------+
|              Kubernetes Cluster (Pods)                |
|  [Component A] --> [Component B] --> [Component C]   |
+------------------------------------------------------+
```

**Key services:**

| Service | Role |
|---|---|
| **KFP API Server** | REST + gRPC API that manages pipelines, runs, experiments |
| **Argo Workflows** | Executes the DAG by scheduling Kubernetes pods |
| **ML Metadata (MLMD)** | Stores artifacts, executions, and lineage in MySQL |
| **Artifact Store** | MinIO (local) or GCS/S3 (cloud) for datasets, models, metrics |
| **KFP UI** | React dashboard to visualise runs, compare metrics, inspect artifacts |

### KFP SDK v2 -- Key Concepts

| Concept | Description |
|---|---|
| **Component** | A single, self-contained step (a Python function or a container image) |
| **Pipeline** | A directed acyclic graph (DAG) of components |
| **Run** | A single execution of a pipeline with specific parameter values |
| **Experiment** | A logical grouping of runs for comparison |
| **Artifact** | A typed output (Dataset, Model, Metrics, HTML, Markdown) persisted to the artifact store |

---

## Hands-On Lab

### Prerequisites Check

```bash
# Python version (3.10+ required)
python --version

# Docker running
docker --version

# pip available
pip --version
```

### Exercise 1: Install the KFP SDK

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install KFP v2
pip install "kfp>=2.0,<3.0"

# Verify installation
python -c "import kfp; print(f'KFP version: {kfp.__version__}')"
```

Expected output:

```
KFP version: 2.x.x
```

### Exercise 2: Your First Pipeline -- Hello KFP

Create a file called `hello_pipeline.py`:

```python
from kfp import dsl, compiler


@dsl.component(base_image="python:3.11-slim")
def say_hello(name: str) -> str:
    """A trivial component that greets the user."""
    greeting = f"Hello, {name}! Welcome to Kubeflow Pipelines."
    print(greeting)
    return greeting


@dsl.component(base_image="python:3.11-slim")
def show_result(message: str) -> None:
    """Print the result of the previous step."""
    print(f"Received from upstream: {message}")


@dsl.pipeline(
    name="hello-pipeline",
    description="A minimal two-step pipeline to verify KFP SDK installation.",
)
def hello_pipeline(recipient: str = "World") -> None:
    hello_task = say_hello(name=recipient)
    show_result(message=hello_task.output)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=hello_pipeline,
        package_path="hello_pipeline.yaml",
    )
    print("Pipeline compiled to hello_pipeline.yaml")
```

Run it:

```bash
python hello_pipeline.py
```

Expected output:

```
Pipeline compiled to hello_pipeline.yaml
```

### Exercise 3: Inspect the Compiled IR

The compiled YAML is the **Intermediate Representation** that the KFP backend executes. Open it and study the structure:

```bash
# View the top-level keys
python -c "
import yaml, json

with open('hello_pipeline.yaml') as f:
    ir = yaml.safe_load(f)

print('Top-level keys:', list(ir.keys()))
print('Pipeline name :', ir.get('pipelineInfo', {}).get('name'))
print('Components    :', list(ir.get('components', {}).keys()))
print('Deployment    :', list(ir.get('deploymentSpec', {}).get('executors', {}).keys()))
"
```

You should see two components (`comp-say-hello`, `comp-show-result`) and two corresponding executors.

### Exercise 4: Understanding Component I/O Types

KFP v2 uses a strong type system for component inputs and outputs. Study the following type mapping:

```python
from kfp import dsl
from kfp.dsl import Dataset, Model, Metrics, Artifact, Input, Output


@dsl.component(base_image="python:3.11-slim")
def demo_types(
    # Parameters (serialised as JSON in the pipeline spec)
    name: str,
    count: int,
    ratio: float,
    flag: bool,
    # Artifact inputs (passed by reference -- the path to the stored file)
    input_data: Input[Dataset],
    # Artifact outputs (KFP provisions the path; you write to it)
    output_data: Output[Dataset],
    output_metrics: Output[Metrics],
) -> str:
    """Demonstrates KFP v2 parameter and artifact types."""
    import pandas as pd

    # Read an upstream artifact
    df = pd.read_csv(input_data.path)

    # Write an output artifact
    df.to_csv(output_data.path, index=False)

    # Log structured metrics
    output_metrics.log_metric("row_count", len(df))
    output_metrics.log_metric("ratio", ratio)

    return f"Processed {len(df)} rows for {name}"
```

**Key takeaway:** Parameters are lightweight values (str, int, float, bool, list, dict). Artifacts are files persisted to the object store and tracked by ML Metadata.

---

## Key Terminology

| Term | Definition |
|---|---|
| **KFP** | Kubeflow Pipelines -- the orchestration platform for ML workflows on Kubernetes |
| **Component** | A self-contained step with defined inputs, outputs, and a container image |
| **Pipeline** | A DAG of components that defines the execution order and data flow |
| **Run** | One execution of a pipeline, identified by a unique run ID |
| **Experiment** | A namespace for grouping and comparing runs |
| **Artifact** | A typed, versioned file (Dataset, Model, Metrics) stored in the artifact store |
| **IR YAML** | The compiled Intermediate Representation of a pipeline |
| **ML Metadata (MLMD)** | The metadata store that tracks artifacts, executions, and lineage |
| **Argo Workflows** | The Kubernetes-native workflow engine that executes KFP pipelines |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Using `kfp` v1 API with v2 docs | `ImportError` or unexpected behaviour | Ensure `kfp>=2.0` and use `@dsl.component` not `@func_to_container_op` |
| Forgetting `packages_to_install` | `ModuleNotFoundError` inside the container | Add all runtime deps to `packages_to_install=[]` |
| Returning a complex object | `TypeError` during compilation | Components can only return primitives (str, int, float, bool) or named tuples |
| Passing artifacts as parameters | Compilation error | Use `Input[Dataset]` / `Output[Model]`, not raw strings |

---

## Self-Check Questions

1. What are the three main services that make up a Kubeflow Pipelines deployment?
2. What is the difference between a *parameter* and an *artifact* in KFP v2?
3. Why does each component run in its own container?
4. What file format does `compiler.Compiler().compile()` produce, and what does the backend do with it?
5. How would you pass a dataset from one component to the next?

---

## You Know You Have Completed This Module When...

- [ ] KFP SDK v2 is installed and `import kfp` works
- [ ] You compiled `hello_pipeline.py` and can read the IR YAML
- [ ] You can explain the KFP architecture diagram from memory
- [ ] You understand the difference between parameters and artifacts
- [ ] Self-check questions answered confidently

---

**Next: [Module 02 -- Kubeflow Installation and Setup -->](../02-kubeflow-setup/)**
