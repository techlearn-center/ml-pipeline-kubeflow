# Capstone Project: Production ML Pipeline with Kubeflow

## Overview

This capstone project combines everything you learned across all 10 modules into a single, production-grade ML pipeline system. You will build an end-to-end pipeline that ingests data, trains multiple models, evaluates them, deploys the best candidate, monitors for drift, and retrains automatically -- all orchestrated by Kubeflow Pipelines on Kubernetes.

This is the project you will showcase to hiring managers.

---

## The Challenge

Build a **complete ML pipeline platform** that demonstrates mastery of every module:

| Module | Capstone requirement |
|---|---|
| 01 - ML Pipelines Overview | Solid KFP v2 architecture with proper component/artifact design |
| 02 - Components | Both function-based and container-based components |
| 03 - Pipeline Definition | Multi-branch DAG with conditional logic and parallelism |
| 04 - Data Processing | Ingestion, validation, and feature engineering steps |
| 05 - Training | Multiple algorithm support (sklearn, XGBoost, at least 2 models) |
| 06 - Evaluation | Metrics-based deployment gate with threshold parameters |
| 07 - Scheduling | Recurring retraining schedule with cron expression |
| 08 - Metadata | Custom metadata on every artifact, MLflow integration |
| 09 - Multi-Step | End-to-end train-evaluate-deploy with notifications |
| 10 - Production | CI/CD workflow, data drift monitoring, unit tests |

---

## Architecture

Your capstone solution should implement this architecture:

```
                  +------------------+
                  |  GitHub Actions  |
                  |  (CI/CD)         |
                  +--------+---------+
                           |
                  +--------v---------+
                  |  Compile + Test  |
                  +--------+---------+
                           |
                  +--------v---------+
                  |  Upload to KFP   |
                  +--------+---------+
                           |
  +------------------------v---------------------------+
  |              Kubeflow Pipelines                     |
  |                                                     |
  |  +--------+   +---------+   +-------+   +--------+ |
  |  | Ingest |-->| Validate|-->| Split |-->| Feature| |
  |  +--------+   +---------+   +-------+   | Eng.   | |
  |                                          +---+----+ |
  |                                              |      |
  |                 +----------------------------+      |
  |                 |                            |      |
  |           +-----v------+            +-------v----+ |
  |           | Train RF   |            | Train XGB  | |
  |           +-----+------+            +------+-----+ |
  |                 |                          |        |
  |           +-----v------+            +-----v------+ |
  |           | Evaluate   |            | Evaluate   | |
  |           +-----+------+            +------+-----+ |
  |                 |                          |        |
  |           +-----v--------------------------v-----+  |
  |           |        Select Best Model             |  |
  |           +----------------+---------------------+  |
  |                            |                        |
  |                   +--------v--------+               |
  |                   | Deploy (if pass)|               |
  |                   +--------+--------+               |
  |                            |                        |
  |                   +--------v--------+               |
  |                   |  Monitor Drift  |               |
  |                   +-----------------+               |
  +-----------------------------------------------------+
                           |
              +------------v------------+
              |  MLflow Tracking Server |
              +-------------------------+
```

---

## Technical Requirements

### Must Have

- [ ] **Data pipeline:** Ingest from at least 2 sources, validate schema, feature-engineer
- [ ] **Training pipeline:** Train at least 2 different algorithms in parallel branches
- [ ] **Evaluation gate:** Compute accuracy, precision, recall, F1, AUC-ROC; only deploy if thresholds are met
- [ ] **Conditional deployment:** Use `dsl.If` to gate the deploy step
- [ ] **Metadata tracking:** Custom metadata on every artifact (dataset version, git commit, timestamp)
- [ ] **Scheduling:** A recurring run configured via `create_recurring_run()`
- [ ] **Notifications:** Alert on both success and failure
- [ ] **CI/CD:** GitHub Actions workflow that lints, tests, compiles, and uploads the pipeline
- [ ] **Unit tests:** At least 5 pytest tests covering component logic
- [ ] **Documentation:** This README with architecture diagram and setup instructions

### Nice to Have

- [ ] Data drift detection pipeline that triggers retraining
- [ ] MLflow integration for experiment tracking and model registry
- [ ] KServe or Seldon deployment component
- [ ] Canary or blue-green deployment strategy
- [ ] GPU training component for a PyTorch model
- [ ] Pipeline versioning with git tags
- [ ] Cost optimization annotations (spot instances, resource right-sizing)

---

## Getting Started

### Step 1: Set Up the Environment

```bash
# Clone the repo
git clone https://github.com/techlearn-center/ml-pipeline-kubeflow.git
cd ml-pipeline-kubeflow

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your values
```

### Step 2: Start Local Infrastructure

```bash
# Start MinIO, MySQL, MLflow, and the pipeline runner
docker compose up -d

# Verify services are running
docker compose ps

# Access the services:
#   MinIO Console:  http://localhost:9001
#   MLflow UI:      http://localhost:5000
#   KFP UI:         http://localhost:8080 (after port-forward)
```

### Step 3: Compile and Run the Pipeline

```bash
# Compile the training pipeline
python -m src.pipelines.training_pipeline

# Inspect the compiled IR
python -c "
import yaml
with open('pipeline.yaml') as f:
    ir = yaml.safe_load(f)
print(f'Pipeline: {ir[\"pipelineInfo\"][\"name\"]}')
print(f'Components: {len(ir[\"components\"])}')
"

# Submit to KFP (requires a running KFP instance)
python -m src.pipelines.training_pipeline --submit
```

### Step 4: Build Your Capstone

Use the source files in `src/` as your starting point:

```
src/
  components/
    data_loader.py      # Data ingestion and splitting
    trainer.py          # Model training (sklearn, XGBoost)
    evaluator.py        # Evaluation and deployment gating
  pipelines/
    training_pipeline.py # End-to-end orchestration
```

Extend these files to meet all the capstone requirements listed above.

### Step 5: Test Your Solution

```bash
# Run unit tests
pytest tests/ -v

# Compile the pipeline (should succeed without errors)
python -m src.pipelines.training_pipeline

# Run the validation script
bash capstone/validation/validate.sh
```

---

## Evaluation Criteria

| Criteria | Weight | What we look for |
|---|---|---|
| **Functionality** | 30% | Pipeline compiles, runs, and produces correct results |
| **Architecture** | 20% | Clean DAG design, proper component separation, reusability |
| **Production readiness** | 20% | CI/CD, monitoring, error handling, notifications |
| **Code quality** | 15% | Type hints, docstrings, tests, linting |
| **Documentation** | 15% | Clear README, architecture diagram, setup instructions |

---

## Sample Pipeline Parameters

When you submit your capstone pipeline, use these parameters as a baseline:

```python
arguments = {
    "dataset_name": "breast_cancer",
    "model_type": "random_forest",
    "n_estimators": 200,
    "max_depth": 7,
    "learning_rate": 0.1,
    "test_size": 0.2,
    "accuracy_threshold": 0.90,
    "f1_threshold": 0.85,
    "model_name": "capstone-classifier",
    "random_state": 42,
}
```

---

## Showcasing to Hiring Managers

When you complete this capstone:

1. **Fork this repo** to your personal GitHub
2. **Add your capstone solution** with clear commit messages
3. **Update this README** with your specific architecture decisions
4. **Include screenshots** of the KFP UI showing successful runs
5. **Record a 3-minute demo video** walking through the pipeline
6. **Reference it on your resume:** "Built a production ML pipeline platform with Kubeflow Pipelines featuring automated training, evaluation gating, conditional deployment, data drift monitoring, and CI/CD -- processing datasets across parallel model training branches with metadata lineage tracking"

### What interviewers will ask about

- "Walk me through the pipeline DAG. Why did you structure it this way?"
- "How does the evaluation gate decide whether to deploy?"
- "What happens if the pipeline fails at the training step?"
- "How would you scale this to handle 100x more data?"
- "How do you detect and respond to data drift?"
- "Explain your CI/CD setup for the pipeline."

---

See [docs/portfolio-guide.md](../docs/portfolio-guide.md) for detailed guidance on presenting this project in interviews.
