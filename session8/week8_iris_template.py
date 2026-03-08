"""session 8 student template: NumPy arrays, centroids, masks, and imports."""

from session8.iris_session8_utils import (
    boolean_mask_for_label,
    class_counts,
    compute_class_centroids,
    load_iris_csv,
)
from session8.iris_session8_models import NumpyCentroidClassifier
import os
import sys

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main():
    print("=== Student Label ===")
    student_id = "YOUR_ID_HERE"  # TODO 1: replace
    full_name = "YOUR_FULL_NAME_HERE"  # TODO 2: replace
    print("Student ID:", student_id)
    print("Full Name:", full_name)

    # TODO 3: load X and y from CSV (or built-in fallback)
    X, y = load_iris_csv("iris.csv", skip_header=1)

    print("\n=== Array Overview ===")
    print("X shape:", X.shape)
    print("X dtype:", X.dtype)
    print("y shape:", y.shape)
    print("y dtype:", y.dtype)

    print("\n=== Indexing and Slicing ===")
    print("First row:", X[0])
    print("Last row:", X[-1])  # TODO 4: keep negative index example
    print("First 5 rows:\n", X[:5])
    print("Petal matrix shape (cols 2:4):", X[:, 2:4].shape)
    print("First 10 labels:", y[:10])

    # TODO 5: boolean mask example
    setosa_mask = boolean_mask_for_label(y, "setosa")
    X_setosa = X[setosa_mask]
    print("Setosa rows via mask:", X_setosa.shape[0])

    print("\n=== Class Counts ===")
    counts = class_counts(y)
    for label, count in counts.items():
        print(label, "->", count)

    print("\n=== Class Centroids ===")
    centroids = compute_class_centroids(X, y)
    for label in sorted(centroids.keys()):
        print(label, "->", np.round(centroids[label], 2))

    print("\n=== Classifier Interface Reuse (from session 7) ===")
    # TODO 6: keep same fit/predict_one/predict_all/evaluate interface
    model = NumpyCentroidClassifier(metric="l2")
    model.fit(X, y)
    result = model.evaluate(X, y)
    print("Correct:", result["correct"])
    print("Wrong:", result["wrong"])
    print("Total:", result["total"])
    print("Accuracy (%):", round(result["accuracy"], 2))

    print("\n=== Metric Comparison (l2 vs l1) ===")
    for metric in ["l2", "l1"]:
        metric_model = NumpyCentroidClassifier(metric=metric)
        metric_model.fit(X, y)
        metric_result = metric_model.evaluate(X, y)
        print(metric, "accuracy (%):", round(metric_result["accuracy"], 2))


if __name__ == "__main__":
    main()
