"""
trainer.py -- KFP v2 component for training ML models.

Supports:
  - scikit-learn classifiers (RandomForest, GradientBoosting, LogisticRegression)
  - XGBoost classifier
  - Hyperparameter configuration via pipeline parameters

The trained model is serialised with joblib and written as a KFP Model artifact.
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Model, Output, Metrics


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas>=2.1",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "numpy>=1.25",
        "joblib>=1.3",
    ],
)
def trainer(
    train_dataset: Input[Dataset],
    model_artifact: Output[Model],
    training_metrics: Output[Metrics],
    model_type: str = "random_forest",
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> None:
    """Train a classification model and persist it as a KFP artifact.

    Args:
        train_dataset: Input CSV artifact from the data_loader component.
        model_artifact: Output model artifact (joblib-serialised).
        training_metrics: Output metrics logged during training.
        model_type: One of "random_forest", "gradient_boosting",
            "logistic_regression", or "xgboost".
        n_estimators: Number of trees / boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Learning rate (gradient boosting / xgboost only).
        random_state: Seed for reproducibility.
    """
    import pandas as pd
    import numpy as np
    import joblib
    import time

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # -----------------------------------------------------------------
    # 1. Load training data
    # -----------------------------------------------------------------
    train_df = pd.read_csv(train_dataset.path)
    target_col = "target"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # -----------------------------------------------------------------
    # 2. Select model
    # -----------------------------------------------------------------
    if model_type == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        estimator = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    elif model_type == "logistic_regression":
        estimator = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )
    elif model_type == "xgboost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
        )
    else:
        raise ValueError(
            f"Unknown model_type='{model_type}'. "
            "Choose from: random_forest, gradient_boosting, logistic_regression, xgboost"
        )

    # Wrap in a sklearn Pipeline with optional scaling
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ])

    print(f"Training {model_type} model ...")

    # -----------------------------------------------------------------
    # 3. Train
    # -----------------------------------------------------------------
    start = time.time()
    pipe.fit(X_train, y_train)
    elapsed = time.time() - start

    train_accuracy = float(pipe.score(X_train, y_train))
    print(f"Training accuracy: {train_accuracy:.4f}  ({elapsed:.1f}s)")

    # -----------------------------------------------------------------
    # 4. Persist model
    # -----------------------------------------------------------------
    model_artifact.metadata["model_type"] = model_type
    model_artifact.metadata["framework"] = "sklearn"
    joblib.dump(pipe, model_artifact.path)
    print(f"Model saved to {model_artifact.path}")

    # -----------------------------------------------------------------
    # 5. Log training metrics
    # -----------------------------------------------------------------
    training_metrics.log_metric("model_type", model_type)
    training_metrics.log_metric("n_estimators", n_estimators)
    training_metrics.log_metric("max_depth", max_depth)
    training_metrics.log_metric("learning_rate", learning_rate)
    training_metrics.log_metric("train_accuracy", train_accuracy)
    training_metrics.log_metric("training_time_sec", round(elapsed, 2))
    training_metrics.log_metric("num_features", int(X_train.shape[1]))
    training_metrics.log_metric("num_train_samples", int(X_train.shape[0]))

    print("Training complete.")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    data = load_breast_cancer()
    X_train, _, y_train, _ = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    print(f"Standalone test -- train accuracy: {clf.score(X_train, y_train):.4f}")
