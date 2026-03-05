# Module 02: Pipeline Components -- Python Function-Based and Container-Based

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Beginner-Intermediate |
| **Prerequisites** | Module 01 completed, KFP SDK installed, Docker running |

---

## Learning Objectives

By the end of this module, you will be able to:

- Create Python function-based components using `@dsl.component`
- Build container-based components from a custom Dockerfile
- Pass parameters and artifacts between components
- Add Python package dependencies to components
- Test components locally before deploying to a cluster

---

## Concepts

### Two Ways to Define Components

KFP v2 supports two component authoring styles. Both produce a containerised step, but they differ in how you specify the environment.

| Style | When to use | Pros | Cons |
|---|---|---|---|
| **Python function-based** | Most ML steps | Fast iteration, no Dockerfile needed | Limited to pip-installable deps |
| **Container-based** | System-level deps, custom runtimes | Full control over the image | Slower iteration (rebuild image) |

### Python Function-Based Components

The `@dsl.component` decorator turns a regular Python function into a KFP component. The function body is extracted, serialised, and executed inside a container at runtime.

```python
from kfp import dsl
from kfp.dsl import Dataset, Output, Metrics


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1", "scikit-learn>=1.3"],
)
def preprocess_data(
    raw_csv_path: str,
    processed_data: Output[Dataset],
    stats: Output[Metrics],
    drop_nulls: bool = True,
) -> None:
    """Load a CSV, clean it, and write a processed artifact."""
    import pandas as pd

    df = pd.read_csv(raw_csv_path)
    initial_rows = len(df)

    if drop_nulls:
        df = df.dropna()

    # Feature engineering example
    if "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df = df.drop(columns=["date"])

    # Write output
    df.to_csv(processed_data.path, index=False)

    # Log stats
    stats.log_metric("initial_rows", initial_rows)
    stats.log_metric("final_rows", len(df))
    stats.log_metric("dropped_rows", initial_rows - len(df))
    stats.log_metric("num_columns", len(df.columns))
```

**Rules for function-based components:**

1. All imports must be *inside* the function body (the function is serialised independently).
2. Only primitive types (str, int, float, bool, list, dict) and KFP artifact types are allowed as parameters.
3. The function must be self-contained -- no references to outer scope variables.

### Container-Based Components

When you need system libraries (e.g., `libgomp` for LightGBM, CUDA for GPU training), build a custom Docker image and reference it:

```python
from kfp import dsl


@dsl.container_component
def train_lightgbm(
    train_path: dsl.InputPath("Dataset"),
    model_path: dsl.OutputPath("Model"),
    num_leaves: int = 31,
    learning_rate: float = 0.05,
):
    """Train a LightGBM model using a custom container image."""
    return dsl.ContainerSpec(
        image="my-registry.io/lightgbm-trainer:v1.2",
        command=["python", "train.py"],
        args=[
            "--train-path", train_path,
            "--model-path", model_path,
            "--num-leaves", str(num_leaves),
            "--learning-rate", str(learning_rate),
        ],
    )
```

The corresponding `Dockerfile` for this image:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install lightgbm pandas scikit-learn

COPY train.py /app/train.py
WORKDIR /app

ENTRYPOINT ["python", "train.py"]
```

---

## Hands-On Lab

### Exercise 1: Build a Data Validation Component

Create `components/data_validator.py`:

```python
from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1", "numpy>=1.25"],
)
def validate_data(
    dataset: Input[Dataset],
    validation_report: Output[Metrics],
    max_null_pct: float = 0.05,
    min_rows: int = 100,
) -> str:
    """Validate a dataset and return PASS or FAIL."""
    import pandas as pd
    import numpy as np

    df = pd.read_csv(dataset.path)
    issues = []

    # Check minimum row count
    if len(df) < min_rows:
        issues.append(f"Row count {len(df)} < minimum {min_rows}")

    # Check null percentage per column
    null_pcts = df.isnull().mean()
    bad_cols = null_pcts[null_pcts > max_null_pct]
    if len(bad_cols) > 0:
        for col, pct in bad_cols.items():
            issues.append(f"Column '{col}' has {pct:.1%} nulls (max {max_null_pct:.1%})")

    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"{dup_count} duplicate rows found")

    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        issues.append(f"{inf_count} infinite values found")

    # Log metrics
    validation_report.log_metric("num_rows", int(len(df)))
    validation_report.log_metric("num_columns", int(len(df.columns)))
    validation_report.log_metric("null_pct_max", float(null_pcts.max()))
    validation_report.log_metric("duplicate_rows", int(dup_count))
    validation_report.log_metric("issues_found", len(issues))

    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        return "FAIL"
    else:
        print("PASS: All validation checks passed")
        return "PASS"
```

### Exercise 2: Build a Feature Engineering Component

```python
from kfp import dsl
from kfp.dsl import Dataset, Input, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1", "scikit-learn>=1.3"],
)
def feature_engineer(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    scaling_method: str = "standard",
    add_polynomial: bool = False,
    poly_degree: int = 2,
) -> None:
    """Apply feature engineering transforms."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

    df = pd.read_csv(input_dataset.path)
    target_col = "target"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scale numeric features
    if scaling_method == "standard":
        scaler = StandardScaler()
    elif scaling_method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")

    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )

    # Optionally add polynomial features
    if add_polynomial:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X_poly = poly.fit_transform(X_scaled)
        col_names = poly.get_feature_names_out(X.columns)
        X_scaled = pd.DataFrame(X_poly, columns=col_names, index=X.index)

    result = pd.concat([X_scaled, y], axis=1)
    result.to_csv(output_dataset.path, index=False)
    print(f"Feature engineering complete: {result.shape}")
```

### Exercise 3: Chain Components into a Mini-Pipeline

```python
from kfp import dsl, compiler

# (Import validate_data and feature_engineer from above)


@dsl.pipeline(name="data-prep-pipeline")
def data_prep_pipeline(
    raw_csv: str = "gs://my-bucket/raw_data.csv",
    max_null_pct: float = 0.05,
) -> None:
    # Step 1: Load raw data
    load_task = load_raw_data(raw_csv_path=raw_csv)

    # Step 2: Validate
    validate_task = validate_data(
        dataset=load_task.outputs["output_dataset"],
        max_null_pct=max_null_pct,
    )

    # Step 3: Feature engineering (only if validation passes)
    with dsl.If(validate_task.output == "PASS"):
        fe_task = feature_engineer(
            input_dataset=load_task.outputs["output_dataset"],
        )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=data_prep_pipeline,
        package_path="data_prep_pipeline.yaml",
    )
    print("Compiled data_prep_pipeline.yaml")
```

### Exercise 4: Local Testing Without a Cluster

You can test component logic locally without KFP. Extract the function body and call it directly:

```python
"""test_components_locally.py -- run component logic without KFP."""
import pandas as pd
from sklearn.datasets import load_iris

# Simulate the data_loader component
data = load_iris(as_frame=True)
df = data.frame
df.to_csv("/tmp/test_dataset.csv", index=False)

# Simulate the feature_engineer logic
from sklearn.preprocessing import StandardScaler

X = df.drop(columns=["target"])
y = df["target"]
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
result = pd.concat([X_scaled, y], axis=1)
print(result.describe())
print(f"\nLocal test passed: {result.shape}")
```

Run it:

```bash
python test_components_locally.py
```

---

## Component Best Practices

| Practice | Why |
|---|---|
| Keep components small and single-purpose | Easier to test, cache, and reuse |
| Pin package versions in `packages_to_install` | Reproducible builds |
| Use `Output[Metrics]` for anything you want to compare across runs | Shows up in the KFP UI comparison view |
| Add `set_display_name()` to every task | Makes the DAG readable in the UI |
| Use `set_caching_options(enable_caching=True)` for expensive steps | Avoids re-running unchanged steps |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Importing at module level in a `@dsl.component` | `ModuleNotFoundError` at runtime | Move all imports inside the function body |
| Returning a DataFrame | `TypeError: unsupported type` | Write to `Output[Dataset]` instead; return only primitives |
| Hardcoding file paths | Works locally, fails on the cluster | Use `Input[Dataset].path` / `Output[Dataset].path` |
| Not listing deps in `packages_to_install` | `ModuleNotFoundError` | List every non-stdlib package |

---

## Self-Check Questions

1. Why must all imports be inside the function body for `@dsl.component`?
2. When would you choose a container-based component over a function-based one?
3. How does KFP pass a `Dataset` artifact from one component to the next?
4. What happens if you enable caching and re-run a pipeline with the same inputs?
5. How can you test component logic locally without a Kubeflow cluster?

---

## You Know You Have Completed This Module When...

- [ ] Created a function-based component with `@dsl.component`
- [ ] Understand `Input[Dataset]` vs `Output[Dataset]` vs primitive parameters
- [ ] Chained two components in a mini-pipeline and compiled it
- [ ] Tested component logic locally without a cluster
- [ ] Can explain when to use container-based vs function-based components

---

**Next: [Module 03 -- Pipeline Components and Containers -->](../03-pipeline-components/)**
