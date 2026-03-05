# Module 05: Training Components -- scikit-learn, XGBoost, and GPU Training

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 04 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Build training components for scikit-learn, XGBoost, and PyTorch models
- Configure GPU resources for deep learning training
- Implement hyperparameter tuning with parallel pipeline branches
- Persist trained models as KFP `Model` artifacts
- Track training metrics with `Output[Metrics]`

---

## Concepts

### Training in a Pipeline Context

In production, model training is never an isolated notebook. It is a step in a larger pipeline that receives processed data from upstream, trains one or more models, and passes the best model to evaluation and deployment downstream.

**Key principles:**

| Principle | Why |
|---|---|
| Reproducibility | Same data + same code + same params = same model |
| Isolation | Each training run executes in its own container |
| Parametrisation | Hyperparameters are pipeline parameters, not hardcoded |
| Artifact tracking | The model file is a versioned KFP `Model` artifact |
| Resource control | GPU, memory, and CPU limits are declared per task |

### sklearn / XGBoost Training Component Pattern

```python
from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output, Metrics


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.1", "scikit-learn>=1.3",
        "xgboost>=2.0", "joblib>=1.3",
    ],
)
def train_model(
    train_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics_out: Output[Metrics],
    algorithm: str = "random_forest",
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> float:
    """Train a classifier and return the training accuracy."""
    import pandas as pd
    import joblib
    import time
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    df = pd.read_csv(train_data.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    estimators = {
        "random_forest": RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_state,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state,
        ),
    }

    if algorithm == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=random_state,
            eval_metric="logloss",
        )
    else:
        clf = estimators[algorithm]

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    start = time.time()
    pipe.fit(X, y)
    elapsed = time.time() - start

    accuracy = pipe.score(X, y)

    # Persist model
    model_artifact.metadata["algorithm"] = algorithm
    model_artifact.metadata["framework"] = "sklearn"
    joblib.dump(pipe, model_artifact.path)

    # Log metrics
    metrics_out.log_metric("algorithm", algorithm)
    metrics_out.log_metric("train_accuracy", round(accuracy, 4))
    metrics_out.log_metric("training_seconds", round(elapsed, 2))
    metrics_out.log_metric("n_estimators", n_estimators)
    metrics_out.log_metric("max_depth", max_depth)

    print(f"{algorithm}: accuracy={accuracy:.4f} in {elapsed:.1f}s")
    return accuracy
```

### PyTorch Training Component (GPU)

For deep learning, use a container-based component with a CUDA base image:

```python
@dsl.component(
    base_image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    packages_to_install=["pandas>=2.1", "scikit-learn>=1.3"],
)
def train_pytorch(
    train_data: Input[Dataset],
    model_artifact: Output[Model],
    metrics_out: Output[Metrics],
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
) -> float:
    """Train a PyTorch neural network classifier."""
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler

    # Load data
    df = pd.read_csv(train_data.path)
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define model
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, n_classes),
            )

        def forward(self, x):
            return self.layers(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor.to(device)).argmax(dim=1).cpu().numpy()
    accuracy = float((preds == y).mean())

    # Save model
    torch.save(model.state_dict(), model_artifact.path)
    model_artifact.metadata["framework"] = "pytorch"

    metrics_out.log_metric("train_accuracy", round(accuracy, 4))
    metrics_out.log_metric("epochs", epochs)
    metrics_out.log_metric("device", str(device))

    print(f"PyTorch training complete: accuracy={accuracy:.4f}")
    return accuracy
```

### Requesting GPU Resources

```python
train_task = train_pytorch(
    train_data=load_task.outputs["train_data"],
    epochs=50,
    learning_rate=0.001,
)
train_task.set_display_name("Train PyTorch (GPU)")
train_task.set_gpu_limit(1)
train_task.set_memory_limit("8Gi")
train_task.set_cpu_limit("4")

# Request a specific GPU type (node selector)
from kfp import kubernetes
kubernetes.add_node_selector(
    train_task,
    label_key="cloud.google.com/gke-accelerator",
    label_value="nvidia-tesla-t4",
)
```

---

## Hands-On Lab

### Exercise 1: Hyperparameter Sweep Pipeline

```python
from kfp import dsl, compiler


@dsl.pipeline(name="hyperparam-sweep")
def hyperparam_sweep(
    dataset_name: str = "breast_cancer",
    algorithms: list = ["random_forest", "gradient_boosting", "xgboost"],
    n_estimators_list: list = [50, 100, 200],
) -> None:
    load_task = data_loader(dataset_name=dataset_name)

    # Sweep over algorithms
    with dsl.ParallelFor(algorithms) as algo:
        # Nested sweep over n_estimators
        with dsl.ParallelFor(n_estimators_list) as n_est:
            train_task = train_model(
                train_data=load_task.outputs["train_data"],
                algorithm=algo,
                n_estimators=n_est,
            )
            train_task.set_display_name(f"Train")
            train_task.set_memory_limit("2Gi")
            train_task.set_caching_options(enable_caching=True)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=hyperparam_sweep,
        package_path="hyperparam_sweep.yaml",
    )
```

### Exercise 2: Training with Cross-Validation

```python
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1", "scikit-learn>=1.3", "numpy>=1.25"],
)
def train_with_cv(
    train_data: dsl.Input[dsl.Dataset],
    metrics_out: dsl.Output[dsl.Metrics],
    algorithm: str = "random_forest",
    n_folds: int = 5,
    n_estimators: int = 100,
) -> float:
    """Train with k-fold cross-validation and return mean CV accuracy."""
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    df = pd.read_csv(train_data.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    if algorithm == "random_forest":
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    scores = cross_val_score(pipe, X, y, cv=n_folds, scoring="accuracy")

    mean_acc = float(scores.mean())
    std_acc = float(scores.std())

    metrics_out.log_metric("cv_mean_accuracy", round(mean_acc, 4))
    metrics_out.log_metric("cv_std_accuracy", round(std_acc, 4))
    metrics_out.log_metric("n_folds", n_folds)
    metrics_out.log_metric("algorithm", algorithm)

    print(f"{algorithm} CV accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")
    return mean_acc
```

### Exercise 3: Model Registry Integration with MLflow

```python
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow>=2.9", "joblib>=1.3", "scikit-learn>=1.3"],
)
def register_model(
    model_artifact: dsl.Input[dsl.Model],
    model_name: str = "sklearn-classifier",
    mlflow_tracking_uri: str = "http://mlflow:5000",
    accuracy: float = 0.0,
) -> str:
    """Register a trained model in MLflow Model Registry."""
    import mlflow
    import joblib

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("kfp-training")

    model = joblib.load(model_artifact.path)

    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        run_id = run.info.run_id
        print(f"Registered model '{model_name}' in run {run_id}")

    return run_id
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Not scaling features before training | Poor convergence for gradient-based models | Use `StandardScaler` in a sklearn `Pipeline` |
| GPU requested but image lacks CUDA | `CUDA not available` | Use a CUDA base image like `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` |
| Large model file exceeds artifact store limits | Upload failure | Compress with joblib, or use cloud storage |
| Hardcoding hyperparameters | Cannot iterate | Make every hyperparameter a pipeline parameter |

---

## Self-Check Questions

1. Why should training hyperparameters be pipeline parameters instead of hardcoded?
2. How do you request a GPU for a KFP component task?
3. What is the advantage of wrapping the model in a sklearn `Pipeline` with a scaler?
4. How would you run a hyperparameter sweep across 3 algorithms and 4 learning rates?
5. Why is cross-validation better than a single train/test split for model selection?

---

## You Know You Have Completed This Module When...

- [ ] Built training components for sklearn and XGBoost
- [ ] Understand how to add a PyTorch GPU training component
- [ ] Implemented a hyperparameter sweep with `dsl.ParallelFor`
- [ ] Tracked training metrics with `Output[Metrics]`
- [ ] Persisted models as KFP `Model` artifacts

---

**Next: [Module 06 -- Model Evaluation Pipelines -->](../06-evaluation-pipelines/)**
