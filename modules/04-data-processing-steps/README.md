# Module 04: Data Processing at Scale -- Parallel Tasks and Caching

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 03 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Build data ingestion, validation, and transformation components
- Process data in parallel using `dsl.ParallelFor`
- Leverage KFP caching to skip unchanged steps
- Handle large datasets efficiently with chunked processing
- Implement data quality gates that halt the pipeline on bad data

---

## Concepts

### Data Processing in ML Pipelines

Production ML pipelines spend most of their time on data, not model training. A typical data processing sub-graph looks like this:

```
  +------------------+
  |  Ingest Raw Data |
  +--------+---------+
           |
  +--------v---------+
  |  Validate Schema |  <-- halt pipeline if schema drift detected
  +--------+---------+
           |
  +--------v---------+
  |  Clean & Impute  |
  +--------+---------+
           |
  +--------v---------+
  | Feature Engineer  |
  +--------+---------+
           |
     +-----+-----+
     |           |
  +--v--+   +---v---+
  |Train|   | Test  |
  +-----+   +-------+
```

### KFP Caching

KFP caches component outputs keyed on: **(component spec + input values + image digest)**. If you re-run a pipeline with the same inputs, cached steps are skipped instantly.

```python
# Enable caching (default is True)
task.set_caching_options(enable_caching=True)

# Disable caching for non-deterministic steps (e.g., data pulls from live DB)
task.set_caching_options(enable_caching=False)
```

**When caching helps:**

| Scenario | Caching? |
|---|---|
| Re-running after a downstream failure | Upstream steps are cached |
| Iterating on training hyperparameters | Data loading is cached |
| Nightly retraining with new data | Cache miss (inputs changed) |
| Debugging a pipeline | Cached steps save time |

### Parallel Data Processing

Use `dsl.ParallelFor` to process data shards, feature groups, or dataset partitions in parallel:

```python
@dsl.pipeline(name="parallel-data-pipeline")
def parallel_data_pipeline(
    partitions: list = ["2024-01", "2024-02", "2024-03", "2024-04"],
) -> None:
    with dsl.ParallelFor(partitions) as partition:
        ingest_task = ingest_partition(partition_key=partition)
        validate_task = validate_partition(
            data=ingest_task.outputs["raw_data"],
        )
        transform_task = transform_partition(
            data=validate_task.outputs["valid_data"],
        )
```

---

## Hands-On Lab

### Exercise 1: Data Ingestion Component

```python
from kfp import dsl
from kfp.dsl import Dataset, Output, Metrics


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1", "scikit-learn>=1.3"],
)
def ingest_data(
    source: str,
    output_data: Output[Dataset],
    ingestion_stats: Output[Metrics],
) -> int:
    """Ingest data from various sources."""
    import pandas as pd
    from sklearn import datasets

    builtin = {
        "breast_cancer": datasets.load_breast_cancer,
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
    }

    if source in builtin:
        bunch = builtin[source]()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        df["target"] = bunch.target
    elif source.startswith("http"):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)

    df.to_csv(output_data.path, index=False)

    ingestion_stats.log_metric("rows", int(len(df)))
    ingestion_stats.log_metric("columns", int(len(df.columns)))
    ingestion_stats.log_metric("size_bytes", int(df.memory_usage(deep=True).sum()))

    print(f"Ingested {len(df)} rows, {len(df.columns)} columns from '{source}'")
    return len(df)
```

### Exercise 2: Schema Validation Component

```python
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1", "numpy>=1.25"],
)
def validate_schema(
    input_data: dsl.Input[dsl.Dataset],
    validated_data: dsl.Output[dsl.Dataset],
    report: dsl.Output[dsl.Metrics],
    expected_columns: str = "",
    max_null_fraction: float = 0.1,
) -> str:
    """Validate data schema and quality. Returns PASS or FAIL."""
    import pandas as pd
    import numpy as np
    import json

    df = pd.read_csv(input_data.path)
    issues = []

    # 1. Check expected columns exist
    if expected_columns:
        expected = json.loads(expected_columns)
        missing = set(expected) - set(df.columns)
        if missing:
            issues.append(f"Missing columns: {missing}")

    # 2. Check null fractions
    null_fracs = df.isnull().mean()
    bad = null_fracs[null_fracs > max_null_fraction]
    for col, frac in bad.items():
        issues.append(f"Column '{col}' null fraction {frac:.2%} > {max_null_fraction:.2%}")

    # 3. Check for all-constant columns (zero variance)
    numeric = df.select_dtypes(include=[np.number])
    zero_var = numeric.columns[numeric.std() == 0].tolist()
    if zero_var:
        issues.append(f"Zero-variance columns: {zero_var}")

    # 4. Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > len(df) * 0.01:
        issues.append(f"{dup_count} duplicate rows ({dup_count/len(df):.1%})")

    # Log metrics
    report.log_metric("num_issues", len(issues))
    report.log_metric("null_fraction_max", float(null_fracs.max()))
    report.log_metric("duplicate_rows", int(dup_count))

    if issues:
        for i in issues:
            print(f"FAIL: {i}")
        return "FAIL"

    df.to_csv(validated_data.path, index=False)
    print("PASS: all schema checks passed")
    return "PASS"
```

### Exercise 3: Chunked Processing for Large Datasets

```python
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1"],
)
def process_in_chunks(
    input_data: dsl.Input[dsl.Dataset],
    output_data: dsl.Output[dsl.Dataset],
    chunk_size: int = 10000,
) -> None:
    """Process a large CSV in chunks to limit memory usage."""
    import pandas as pd

    chunks = []
    for i, chunk in enumerate(pd.read_csv(input_data.path, chunksize=chunk_size)):
        # Apply transforms per chunk
        numeric_cols = chunk.select_dtypes(include="number").columns
        chunk[numeric_cols] = chunk[numeric_cols].fillna(chunk[numeric_cols].median())
        chunks.append(chunk)
        print(f"Processed chunk {i}: {len(chunk)} rows")

    result = pd.concat(chunks, ignore_index=True)
    result.to_csv(output_data.path, index=False)
    print(f"Total: {len(result)} rows after chunked processing")
```

### Exercise 4: Complete Data Processing Pipeline

```python
from kfp import dsl, compiler


@dsl.pipeline(
    name="data-processing-pipeline",
    description="Ingest, validate, clean, and feature-engineer a dataset.",
)
def data_processing_pipeline(
    data_source: str = "breast_cancer",
    max_null_fraction: float = 0.1,
    test_size: float = 0.2,
) -> None:
    # Step 1: Ingest
    ingest_task = ingest_data(source=data_source)
    ingest_task.set_display_name("Ingest Data")
    ingest_task.set_caching_options(enable_caching=True)

    # Step 2: Validate
    validate_task = validate_schema(
        input_data=ingest_task.outputs["output_data"],
        max_null_fraction=max_null_fraction,
    )
    validate_task.set_display_name("Validate Schema")

    # Step 3: Only continue if validation passes
    with dsl.If(validate_task.output == "PASS"):
        clean_task = clean_and_impute(
            input_data=ingest_task.outputs["output_data"],
        )
        clean_task.set_display_name("Clean & Impute")
        clean_task.set_caching_options(enable_caching=True)

        split_task = split_data(
            input_data=clean_task.outputs["cleaned_data"],
            test_size=test_size,
        )
        split_task.set_display_name("Train/Test Split")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=data_processing_pipeline,
        package_path="data_processing_pipeline.yaml",
    )
    print("Compiled data_processing_pipeline.yaml")
```

---

## Caching Deep Dive

### How the Cache Key is Computed

```
cache_key = hash(
    component_spec,       # function code + packages_to_install
    input_parameters,     # all scalar inputs
    input_artifact_hashes,# content hash of input artifacts
    container_image,      # full image digest
)
```

### Controlling Cache Behaviour

```python
# Per-task control
task.set_caching_options(enable_caching=True)

# Force re-execution by changing a "cache-busting" parameter
import time

@dsl.pipeline(name="cache-busting-example")
def pipeline(bust: str = "") -> None:
    task = my_component(cache_buster=bust)
    # Submit with bust=str(time.time()) to force fresh execution
```

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Caching a step that reads from a live database | Stale results | `set_caching_options(enable_caching=False)` |
| Loading entire large file into memory | OOMKilled pod | Use chunked processing or increase memory limit |
| No validation gate | Bad data silently propagates | Add a `validate_schema` step before processing |
| Hardcoded column names | Breaks when schema changes | Pass expected columns as a pipeline parameter |

---

## Self-Check Questions

1. How does KFP decide whether to use a cached result or re-execute a component?
2. When should you disable caching?
3. How would you process a 50 GB CSV file in a KFP component?
4. What is the purpose of a data validation gate in a pipeline?
5. How does `dsl.ParallelFor` help with data processing at scale?

---

## You Know You Have Completed This Module When...

- [ ] Built an ingestion component that loads data from multiple sources
- [ ] Implemented a schema validation gate
- [ ] Used chunked processing for large datasets
- [ ] Controlled caching behaviour for deterministic and non-deterministic steps
- [ ] Wired the data processing steps into a complete pipeline

---

**Next: [Module 05 -- Training Pipelines -->](../05-training-pipelines/)**
