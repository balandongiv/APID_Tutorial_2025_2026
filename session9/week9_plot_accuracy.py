"""session 9 plotting helpers: reuse Session 8 arrays/centroids/evaluation outputs."""

from __future__ import annotations
from session8.iris_session8_utils import compute_class_centroids, load_iris_csv
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

import matplotlib

matplotlib.use("Agg")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _load_results_dataframe():
    xlsx_path = "session9/iris_experiment_results.xlsx"
    csv_path = "session9/iris_experiment_results.csv"

    if os.path.exists(xlsx_path):
        try:
            return pd.read_excel(xlsx_path)
        except Exception:
            pass

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        "No results file found. Run session9/session9_iris_template.py first.")


def plot_scatter_with_centroids(output_path="session9/session9_scatter_centroids.png"):
    """Plot petal_length vs petal_width with centroid overlay."""
    X, y = load_iris_csv("iris.csv", skip_header=1)
    centroids = compute_class_centroids(X, y)

    # Columns 2 and 3 are petal_length and petal_width
    x_col = 2
    y_col = 3

    plt.figure(figsize=(7, 5))
    for label in sorted(set(y.tolist())):
        mask = y == label
        plt.scatter(X[mask, x_col], X[mask, y_col], label=label, alpha=0.75)

    for label, centroid in centroids.items():
        plt.scatter(
            centroid[x_col],
            centroid[y_col],
            marker="X",
            s=180,
            edgecolor="black",
            linewidth=1.0,
            label=f"{label} centroid",
        )

    plt.title("Iris Petal Scatter with Session 8 Centroids")
    plt.xlabel("petal_length")
    plt.ylabel("petal_width")
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print("Saved:", output_path)


def plot_accuracy_from_results(
    boxplot_path="session9/session9_accuracy_boxplot.png",
    lineplot_path="session9/session9_accuracy_lineplot.png",
):
    """Plot accuracy distribution and trend from session 9 experiment results."""
    df = _load_results_dataframe()

    plt.figure(figsize=(7, 5))
    df.boxplot(column="accuracy", by="metric")
    plt.title("Accuracy Distribution by Metric")
    plt.suptitle("")
    plt.xlabel("metric")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(boxplot_path, dpi=150)
    plt.close()
    print("Saved:", boxplot_path)

    avg = df.groupby(["metric", "test_size"])["accuracy"].mean().reset_index()

    plt.figure(figsize=(7, 5))
    for metric in sorted(avg["metric"].unique()):
        sub = avg[avg["metric"] == metric]
        plt.plot(sub["test_size"], sub["accuracy"], marker="o", label=metric)

    plt.title("Mean Accuracy vs Test Size")
    plt.xlabel("test_size")
    plt.ylabel("mean accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(lineplot_path, dpi=150)
    plt.close()
    print("Saved:", lineplot_path)


def generate_all_plots():
    """Generate all session 9 plots."""
    plot_scatter_with_centroids()
    plot_accuracy_from_results()


if __name__ == "__main__":
    generate_all_plots()
