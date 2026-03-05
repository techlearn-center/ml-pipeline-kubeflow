"""
data_loader.py -- KFP v2 component for loading and splitting datasets.

This component:
  1. Loads a dataset from a CSV file, URL, or sklearn built-in dataset.
  2. Performs a stratified train/test split.
  3. Writes the resulting DataFrames as output artifacts so downstream
     components can consume them.

Usage inside a pipeline:
    load_task = data_loader(
        dataset_name="breast_cancer",
        test_size=0.2,
        random_state=42,
    )
"""

from kfp import dsl
from kfp.dsl import Dataset, Output, Artifact, Metrics


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas>=2.1", "scikit-learn>=1.3", "numpy>=1.25"],
)
def data_loader(
    dataset_name: str,
    test_size: float,
    random_state: int,
    train_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    data_stats: Output[Metrics],
) -> None:
    """Load a dataset, split it, and write train/test artifacts.

    Args:
        dataset_name: Name of a sklearn toy dataset (e.g. "breast_cancer",
            "iris", "wine") or a URL/local path to a CSV file.
        test_size: Fraction of data reserved for the test set (0.0 - 1.0).
        random_state: Seed for reproducible splitting.
        train_dataset: Output artifact -- training split as CSV.
        test_dataset: Output artifact -- test split as CSV.
        data_stats: Output metrics -- basic statistics about the dataset.
    """
    import pandas as pd
    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    # -----------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------
    sklearn_datasets = {
        "breast_cancer": datasets.load_breast_cancer,
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "digits": datasets.load_digits,
    }

    if dataset_name in sklearn_datasets:
        bunch = sklearn_datasets[dataset_name]()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        df["target"] = bunch.target
        print(f"Loaded sklearn dataset '{dataset_name}': {df.shape}")
    elif dataset_name.startswith("http"):
        df = pd.read_csv(dataset_name)
        print(f"Loaded remote CSV '{dataset_name}': {df.shape}")
    else:
        df = pd.read_csv(dataset_name)
        print(f"Loaded local CSV '{dataset_name}': {df.shape}")

    # -----------------------------------------------------------------
    # 2. Stratified train / test split
    # -----------------------------------------------------------------
    target_col = "target"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print(f"Train set: {train_df.shape}, Test set: {test_df.shape}")

    # -----------------------------------------------------------------
    # 3. Write output artifacts
    # -----------------------------------------------------------------
    train_df.to_csv(train_dataset.path, index=False)
    test_df.to_csv(test_dataset.path, index=False)

    # -----------------------------------------------------------------
    # 4. Log dataset statistics
    # -----------------------------------------------------------------
    data_stats.log_metric("num_samples", int(len(df)))
    data_stats.log_metric("num_features", int(X.shape[1]))
    data_stats.log_metric("num_classes", int(y.nunique()))
    data_stats.log_metric("train_samples", int(len(train_df)))
    data_stats.log_metric("test_samples", int(len(test_df)))
    data_stats.log_metric("test_size", test_size)
    data_stats.log_metric("class_balance_min", float(y.value_counts(normalize=True).min()))
    data_stats.log_metric("class_balance_max", float(y.value_counts(normalize=True).max()))

    print("Data loading and splitting complete.")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn import datasets
    import pandas as pd

    bunch = datasets.load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    print(f"Standalone test -- loaded {df.shape[0]} rows, {df.shape[1]} cols")
    print(df.head())
