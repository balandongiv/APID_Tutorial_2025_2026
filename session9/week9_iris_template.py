"""session 9 student template: experiment pipeline + plotting preparation (reusing Session 8 modules)."""

from __future__ import annotations
from session9.iris_eval import accuracy_score, train_test_split
from session8.iris_session8_utils import compute_class_centroids, load_iris_csv
from session8.iris_session8_models import NumpyCentroidClassifier

import os
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def run_experiment(test_size=0.2, seed=1, metric="l2"):
    """Run one experiment and return summary dictionary."""
    # TODO 1: load arrays produced in Session 8 style
    X, y = load_iris_csv("iris.csv", skip_header=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, seed=seed)

    # TODO 2: reuse Session 8 classifier interface
    model = NumpyCentroidClassifier(metric=metric)
    model.fit(X_train, y_train)
    y_pred = model.predict_all(X_test)

    acc = accuracy_score(y_test, y_pred)
    centroids = compute_class_centroids(X_train, y_train)

    return {
        "metric": metric,
        "test_size": test_size,
        "seed": seed,
        "accuracy": acc,
        "centroids": centroids,
    }


def run_multiple_trials(metrics, test_sizes, num_trials):
    """Run repeated trials and return list-of-dicts results."""
    results = []

    # TODO 3: loop over settings and trials
    for metric in metrics:
        for test_size in test_sizes:
            for trial in range(1, num_trials + 1):
                run = run_experiment(test_size=test_size,
                                     seed=trial, metric=metric)
                results.append(
                    {
                        "metric": run["metric"],
                        "test_size": run["test_size"],
                        "trial": trial,
                        "seed": run["seed"],
                        "accuracy": run["accuracy"],
                    }
                )
                print(
                    f"metric={metric} | test_size={test_size} | "
                    f"trial={trial} | seed={trial} | acc={run['accuracy']:.3f}"
                )

    return results


def save_results(results, csv_path="session9/iris_experiment_results.csv", xlsx_path="session9/iris_experiment_results.xlsx"):
    """Save results table to CSV and Excel (if Excel writer is available)."""
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    try:
        df.to_excel(xlsx_path, index=False)
        print("Saved:", xlsx_path)
    except Exception as exc:
        print("Excel save skipped:", exc)

    return df


def main():
    print("=== Student Label ===")
    student_id = "YOUR_ID_HERE"  # TODO 4: replace
    full_name = "YOUR_FULL_NAME_HERE"  # TODO 5: replace
    print("Student ID:", student_id)
    print("Full Name:", full_name)

    # TODO 6: keep this experiment grid, or adjust in your own trials
    all_results = run_multiple_trials(
        metrics=["l1", "l2"],
        test_sizes=[0.2, 0.3, 0.4],
        num_trials=5,
    )

    df = save_results(all_results)
    print("\n=== Results Preview ===")
    print(df.head())

    avg = df.groupby(["metric", "test_size"])["accuracy"].mean().reset_index()
    print("\n=== Mean Accuracy by Metric/Test Size ===")
    print(avg)

    print("\nRun this next to generate figures:")
    print("python session9/session9_plot_accuracy.py")


if __name__ == "__main__":
    main()
