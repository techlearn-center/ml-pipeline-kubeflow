"""
evaluator.py -- KFP v2 component for model evaluation and deployment gating.

This component:
  1. Loads the trained model artifact.
  2. Runs predictions on the held-out test set.
  3. Computes classification metrics (accuracy, precision, recall, F1, AUC-ROC).
  4. Compares metrics against configurable thresholds.
  5. Outputs a deploy_decision string ("deploy" or "no_deploy") so downstream
     pipeline steps can conditionally promote the model.
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Metrics, Output


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.1",
        "scikit-learn>=1.3",
        "numpy>=1.25",
        "joblib>=1.3",
    ],
)
def evaluator(
    test_dataset: Input[Dataset],
    model_artifact: Input[Model],
    eval_metrics: Output[Metrics],
    accuracy_threshold: float = 0.85,
    f1_threshold: float = 0.80,
) -> str:
    """Evaluate a trained model and decide whether to deploy.

    Args:
        test_dataset: Held-out test CSV from data_loader.
        model_artifact: Trained model (joblib) from trainer.
        eval_metrics: Output metrics artifact.
        accuracy_threshold: Minimum accuracy to approve deployment.
        f1_threshold: Minimum weighted F1 to approve deployment.

    Returns:
        "deploy" if the model passes all thresholds, otherwise "no_deploy".
    """
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        classification_report,
        confusion_matrix,
    )

    # -----------------------------------------------------------------
    # 1. Load test data and model
    # -----------------------------------------------------------------
    test_df = pd.read_csv(test_dataset.path)
    target_col = "target"
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    model = joblib.load(model_artifact.path)

    print(f"Evaluating on {len(X_test)} test samples ...")

    # -----------------------------------------------------------------
    # 2. Predict
    # -----------------------------------------------------------------
    y_pred = model.predict(X_test)

    # Probabilities for AUC-ROC (handles binary and multiclass)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None

    # -----------------------------------------------------------------
    # 3. Compute metrics
    # -----------------------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # AUC-ROC
    n_classes = len(np.unique(y_test))
    if y_proba is not None and n_classes == 2:
        auc_roc = roc_auc_score(y_test, y_proba[:, 1])
    elif y_proba is not None:
        auc_roc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="weighted"
        )
    else:
        auc_roc = 0.0

    # Print detailed report
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("--- Confusion Matrix ---")
    print(cm)

    # -----------------------------------------------------------------
    # 4. Log metrics
    # -----------------------------------------------------------------
    eval_metrics.log_metric("accuracy", round(accuracy, 4))
    eval_metrics.log_metric("precision", round(precision, 4))
    eval_metrics.log_metric("recall", round(recall, 4))
    eval_metrics.log_metric("f1_score", round(f1, 4))
    eval_metrics.log_metric("auc_roc", round(auc_roc, 4))
    eval_metrics.log_metric("accuracy_threshold", accuracy_threshold)
    eval_metrics.log_metric("f1_threshold", f1_threshold)
    eval_metrics.log_metric("num_test_samples", int(len(X_test)))

    # -----------------------------------------------------------------
    # 5. Deployment decision
    # -----------------------------------------------------------------
    passes_accuracy = accuracy >= accuracy_threshold
    passes_f1 = f1 >= f1_threshold

    if passes_accuracy and passes_f1:
        deploy_decision = "deploy"
        print(
            f"\nDEPLOY -- accuracy={accuracy:.4f} >= {accuracy_threshold}, "
            f"f1={f1:.4f} >= {f1_threshold}"
        )
    else:
        deploy_decision = "no_deploy"
        reasons = []
        if not passes_accuracy:
            reasons.append(f"accuracy={accuracy:.4f} < {accuracy_threshold}")
        if not passes_f1:
            reasons.append(f"f1={f1:.4f} < {f1_threshold}")
        print(f"\nNO DEPLOY -- {', '.join(reasons)}")

    eval_metrics.log_metric("deploy_decision", deploy_decision)
    return deploy_decision


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"Standalone test -- accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Standalone test -- f1:       {f1_score(y_test, y_pred, average='weighted'):.4f}")
