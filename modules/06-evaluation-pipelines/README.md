# Module 06: Model Evaluation and Validation

| | |
|---|---|
| **Time** | 3-5 hours |
| **Difficulty** | Intermediate |
| **Prerequisites** | Module 05 completed |

---

## Learning Objectives

By the end of this module, you will be able to:

- Build evaluation components that compute classification and regression metrics
- Implement deployment gates based on metric thresholds
- Compare candidate models against a production baseline
- Generate evaluation reports with confusion matrices and ROC curves
- Use KFP `Metrics` and `ClassificationMetrics` artifact types

---

## Concepts

### Why Evaluation is a Pipeline Step

Evaluation must be automated and auditable. If evaluation is manual, it becomes a bottleneck. If it is automated but not gated, bad models slip into production.

```
  +----------+     +----------+     +----------+     +----------+
  |  Train   | --> | Evaluate | --> |  Gate    | --> |  Deploy  |
  +----------+     +----------+     +----+-----+     +----------+
                                         |
                                    FAIL |
                                         v
                                   +-----------+
                                   | Alert +   |
                                   | Retrain   |
                                   +-----------+
```

**Evaluation pipeline responsibilities:**

| Responsibility | Detail |
|---|---|
| Compute metrics | Accuracy, precision, recall, F1, AUC-ROC, MSE, etc. |
| Compare to baseline | Is the new model better than what is currently in production? |
| Check for bias | Per-group metrics to detect fairness issues |
| Gate deployment | Only promote models that meet all thresholds |
| Log everything | Metrics, confusion matrix, and plots stored as artifacts |

### KFP Metrics Artifact Types

```python
from kfp.dsl import Metrics, ClassificationMetrics, Output

# Scalar metrics (shown in the KFP UI run comparison table)
metrics: Output[Metrics]
metrics.log_metric("accuracy", 0.95)
metrics.log_metric("f1_score", 0.93)

# Classification-specific (renders confusion matrix and ROC in UI)
classification_metrics: Output[ClassificationMetrics]
classification_metrics.log_confusion_matrix(
    categories=["benign", "malignant"],
    matrix=[[50, 3], [2, 59]],
)
classification_metrics.log_roc_curve(
    fpr=[0.0, 0.1, 0.2, 1.0],
    tpr=[0.0, 0.8, 0.95, 1.0],
    threshold=[1.0, 0.8, 0.5, 0.0],
)
```

---

## Hands-On Lab

### Exercise 1: Comprehensive Evaluation Component

```python
from kfp import dsl
from kfp.dsl import Dataset, Model, Metrics, ClassificationMetrics, Input, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.1", "scikit-learn>=1.3",
        "numpy>=1.25", "joblib>=1.3",
    ],
)
def evaluate_model(
    test_data: Input[Dataset],
    model_artifact: Input[Model],
    scalar_metrics: Output[Metrics],
    classification_report: Output[ClassificationMetrics],
    accuracy_threshold: float = 0.85,
    f1_threshold: float = 0.80,
) -> str:
    """Evaluate a model and return deploy/no_deploy decision."""
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, roc_curve,
    )

    df = pd.read_csv(test_data.path)
    X_test = df.drop(columns=["target"])
    y_test = df["target"].values

    model = joblib.load(model_artifact.path)
    y_pred = model.predict(X_test)

    # --- Scalar metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    scalar_metrics.log_metric("accuracy", round(acc, 4))
    scalar_metrics.log_metric("precision", round(prec, 4))
    scalar_metrics.log_metric("recall", round(rec, 4))
    scalar_metrics.log_metric("f1_score", round(f1, 4))

    # --- AUC-ROC (binary only) ---
    classes = np.unique(y_test)
    if len(classes) == 2 and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        scalar_metrics.log_metric("auc_roc", round(auc, 4))

        # ROC curve for KFP UI
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        classification_report.log_roc_curve(
            fpr=fpr.tolist(),
            tpr=tpr.tolist(),
            threshold=thresholds.tolist(),
        )

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    labels = [str(c) for c in classes]
    classification_report.log_confusion_matrix(
        categories=labels,
        matrix=cm.tolist(),
    )

    # --- Deployment gate ---
    passes = acc >= accuracy_threshold and f1 >= f1_threshold
    decision = "deploy" if passes else "no_deploy"
    scalar_metrics.log_metric("deploy_decision", decision)

    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f} -> {decision}")
    return decision
```

### Exercise 2: Baseline Comparison Component

```python
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.1", "scikit-learn>=1.3",
        "numpy>=1.25", "joblib>=1.3",
    ],
)
def compare_to_baseline(
    test_data: dsl.Input[dsl.Dataset],
    candidate_model: dsl.Input[dsl.Model],
    baseline_model: dsl.Input[dsl.Model],
    comparison_metrics: dsl.Output[dsl.Metrics],
    min_improvement: float = 0.01,
) -> str:
    """Compare candidate model to baseline. Return 'promote' or 'keep_baseline'."""
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, f1_score

    df = pd.read_csv(test_data.path)
    X = df.drop(columns=["target"])
    y = df["target"]

    candidate = joblib.load(candidate_model.path)
    baseline = joblib.load(baseline_model.path)

    # Evaluate both
    cand_acc = accuracy_score(y, candidate.predict(X))
    base_acc = accuracy_score(y, baseline.predict(X))
    cand_f1 = f1_score(y, candidate.predict(X), average="weighted")
    base_f1 = f1_score(y, baseline.predict(X), average="weighted")

    improvement_acc = cand_acc - base_acc
    improvement_f1 = cand_f1 - base_f1

    comparison_metrics.log_metric("candidate_accuracy", round(cand_acc, 4))
    comparison_metrics.log_metric("baseline_accuracy", round(base_acc, 4))
    comparison_metrics.log_metric("accuracy_improvement", round(improvement_acc, 4))
    comparison_metrics.log_metric("candidate_f1", round(cand_f1, 4))
    comparison_metrics.log_metric("baseline_f1", round(base_f1, 4))
    comparison_metrics.log_metric("f1_improvement", round(improvement_f1, 4))

    if improvement_acc >= min_improvement and improvement_f1 >= min_improvement:
        print(f"PROMOTE: candidate is better by acc={improvement_acc:+.4f}, f1={improvement_f1:+.4f}")
        return "promote"
    else:
        print(f"KEEP BASELINE: improvement too small acc={improvement_acc:+.4f}, f1={improvement_f1:+.4f}")
        return "keep_baseline"
```

### Exercise 3: Fairness Check Component

```python
@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1", "scikit-learn>=1.3", "joblib>=1.3"],
)
def fairness_check(
    test_data: dsl.Input[dsl.Dataset],
    model_artifact: dsl.Input[dsl.Model],
    fairness_metrics: dsl.Output[dsl.Metrics],
    sensitive_column: str = "",
    max_disparity: float = 0.1,
) -> str:
    """Check per-group accuracy disparity. Return PASS or FAIL."""
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(test_data.path)
    model = joblib.load(model_artifact.path)

    if not sensitive_column or sensitive_column not in df.columns:
        print("No sensitive column specified; skipping fairness check")
        return "PASS"

    X = df.drop(columns=["target"])
    y = df["target"]
    y_pred = model.predict(X)

    groups = df[sensitive_column].unique()
    group_accuracies = {}

    for group in groups:
        mask = df[sensitive_column] == group
        if mask.sum() < 10:
            continue
        group_acc = accuracy_score(y[mask], y_pred[mask])
        group_accuracies[str(group)] = group_acc
        fairness_metrics.log_metric(f"accuracy_group_{group}", round(group_acc, 4))

    if len(group_accuracies) < 2:
        return "PASS"

    accs = list(group_accuracies.values())
    disparity = max(accs) - min(accs)
    fairness_metrics.log_metric("max_disparity", round(disparity, 4))

    if disparity > max_disparity:
        print(f"FAIL: disparity {disparity:.4f} > {max_disparity}")
        return "FAIL"

    print(f"PASS: disparity {disparity:.4f} within threshold")
    return "PASS"
```

### Exercise 4: Evaluation Pipeline

```python
from kfp import dsl, compiler


@dsl.pipeline(name="evaluation-pipeline")
def evaluation_pipeline(
    dataset_name: str = "breast_cancer",
    accuracy_threshold: float = 0.90,
    f1_threshold: float = 0.85,
) -> None:
    # Load and split
    load_task = data_loader(dataset_name=dataset_name, test_size=0.2)

    # Train candidate
    train_task = train_model(
        train_data=load_task.outputs["train_data"],
        algorithm="random_forest",
    )

    # Evaluate
    eval_task = evaluate_model(
        test_data=load_task.outputs["test_data"],
        model_artifact=train_task.outputs["model_artifact"],
        accuracy_threshold=accuracy_threshold,
        f1_threshold=f1_threshold,
    )
    eval_task.set_display_name("Evaluate Model")

    # Conditional deployment
    with dsl.If(eval_task.output == "deploy"):
        deploy_task = deploy_model(
            model=train_task.outputs["model_artifact"],
        )
        deploy_task.set_display_name("Deploy to Production")

    with dsl.Else():
        alert_task = send_alert(
            message="Model did not pass evaluation thresholds",
        )
        alert_task.set_display_name("Send Alert")


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=evaluation_pipeline,
        package_path="evaluation_pipeline.yaml",
    )
```

---

## Evaluation Best Practices

| Practice | Why |
|---|---|
| Use multiple metrics, not just accuracy | Accuracy is misleading for imbalanced datasets |
| Always compare to a baseline | Absolute thresholds drift; relative comparison is more robust |
| Check per-group fairness | Regulatory and ethical requirement |
| Log confusion matrix as an artifact | Visual inspection catches issues numbers miss |
| Make thresholds pipeline parameters | Different use cases need different thresholds |

---

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Only checking accuracy | Deployed model performs poorly on minority class | Add precision, recall, F1, and per-class metrics |
| No baseline comparison | New model is worse than production but meets threshold | Add `compare_to_baseline` step |
| Evaluating on training data | Inflated metrics | Always evaluate on a held-out test set |
| Hardcoding threshold | Cannot adjust without code change | Make it a pipeline parameter |

---

## Self-Check Questions

1. Why should the evaluation step be a separate component from training?
2. What metrics would you use for a binary classifier vs a multi-class classifier?
3. How does comparing to a baseline prevent model regression?
4. What is the purpose of a fairness check in an ML pipeline?
5. How do `ClassificationMetrics` artifacts render in the KFP UI?

---

## You Know You Have Completed This Module When...

- [ ] Built an evaluation component with accuracy, precision, recall, F1, and AUC-ROC
- [ ] Implemented a deployment gate based on metric thresholds
- [ ] Created a baseline comparison component
- [ ] Logged confusion matrix and ROC curve artifacts
- [ ] Wired evaluation into a pipeline with conditional deployment

---

**Next: [Module 07 -- Model Deployment Pipelines -->](../07-deployment-pipelines/)**
