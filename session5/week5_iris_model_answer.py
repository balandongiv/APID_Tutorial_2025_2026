"""Module convener model answer for session 5 (positional/keyword arguments and fit).

This file is intentionally written as a progression from Session 4.
The classifier rule itself is unchanged, but several function signatures and call
patterns are extended so the script behaves like a small configurable API.

Main Session 5 deltas compared to Session 4:
1) import a helper from another module (session5/helper/iris_utils.py),
2) add optional/default arguments to selected functions,
3) allow positional and keyword override calls from main(),
4) add an optional fit-based threshold path (use_fit=True).
"""

import csv

from helper.iris_utils import fit_threshold_from_setosa


def make_print_status(status_text):
    """Print a status line for user feedback."""
    print(f"[STATUS] {status_text}")


def determine_binary_label(sample, settings):
    """Return prediction label using threshold rule."""
    if sample[settings["feature_name"]] < settings["threshold"]:
        return settings["positive_label"]
    return settings["negative_label"]


def determine_true_binary_label(sample, settings):
    """Convert the sample's true species into a binary label."""
    if sample[settings["label_key"]] == settings["positive_label"]:
        return settings["positive_label"]
    return settings["negative_label"]


def evaluate_prediction(y_pred, y_true):
    """Return True if prediction matches truth, else False."""
    return y_pred == y_true


def classify_sample(sample, settings):
    """Classify one sample and return prediction."""
    return determine_binary_label(sample, settings)


def load_iris_data(filepath=None):
    """Load Iris dataset from CSV with optional default-path fallback.

    Session 4 version required a positional filepath argument.
    Session 5 version changes the signature to filepath=None so beginners can
    run a default workflow without passing arguments, while still allowing
    advanced callers to override the file path.

    Behavior change from Session 4:
    - If filepath is None, print a friendly warning and use
      "session5/iris_data.csv".
    - If filepath is provided, use it directly.
    """
    if filepath is None:
        print("[WARNING] No filepath provided. Using default 'session5/iris_data.csv'.")
        filepath = "session5/iris_data.csv"

    dataset = []
    with open(filepath, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            dataset.append(
                {
                    "id": row["id"],
                    "sepal_length": float(row["sepal_length"]),
                    "sepal_width": float(row["sepal_width"]),
                    "petal_length": float(row["petal_length"]),
                    "petal_width": float(row["petal_width"]),
                    "species": row["species"],
                }
            )
    return dataset


def get_default_settings():
    """Return the default rule settings."""
    return {
        "threshold": 2.0,
        "feature_name": "petal_length",
        "positive_label": "setosa",
        "negative_label": "not_setosa",
        "label_key": "species",
    }


def update_metrics(metrics, y_pred, y_true):
    """Update correct/wrong/total counters and append predictions."""
    is_correct = evaluate_prediction(y_pred, y_true)
    if is_correct:
        metrics["correct"] += 1
    else:
        metrics["wrong"] += 1

    metrics["total"] += 1
    metrics["y_pred_list"].append(y_pred)


def initialize_predictions(dataset, settings, print_each=True):
    """Run prediction loop with optional per-sample output control.

    Session 4 always printed one line per sample.
    Session 5 introduces print_each=True as an optional keyword argument so
    callers can silence verbose logs during experiments while keeping the
    evaluation logic exactly the same.
    """
    metrics = {"correct": 0, "wrong": 0, "total": 0, "y_pred_list": []}

    for sample in dataset:
        y_pred = classify_sample(sample, settings)
        y_true = determine_true_binary_label(sample, settings)
        update_metrics(metrics, y_pred, y_true)

        if print_each:
            print(
                f"id={sample['id']} | true={y_true} | pred={y_pred} | "
                f"petal_length={sample['petal_length']}"
            )

    accuracy = (metrics["correct"] / metrics["total"]) * \
        100 if metrics["total"] else 0.0
    metrics["accuracy"] = accuracy
    return metrics


def setup_application(filepath=None, threshold=None, print_each=True, use_fit=False):
    """Load data, configure settings, and run the pipeline with overrides.

    This is the main Session 5 extension point compared to Session 4.

    Session 4 accepted only filepath and always used fixed defaults.
    Session 5 adds API-style controls:
    - filepath=None: optional path override for dataset loading.
    - threshold=None: optional manual threshold override.
    - print_each=True: optional verbose control for per-sample prints.
    - use_fit=False: optional path to learn threshold from data using
      fit_threshold_from_setosa() imported from helper.iris_utils.

    Priority logic is explicit:
    1) if use_fit is True, learned threshold wins;
    2) else if threshold is provided, manual threshold is used;
    3) else keep default threshold from get_default_settings().
    """
    dataset = load_iris_data(filepath)
    settings = get_default_settings()

    if use_fit:
        settings["threshold"] = fit_threshold_from_setosa(
            dataset, settings["threshold"])
    elif threshold is not None:
        settings["threshold"] = threshold

    result = initialize_predictions(dataset, settings, print_each=print_each)
    return settings, dataset, result


def print_summary(title, result):
    """Print metric summary for one run."""
    print(f"\n=== {title} ===")
    print("Correct:", result["correct"])
    print("Wrong:", result["wrong"])
    print("Total:", result["total"])
    print("Accuracy (%):", round(result["accuracy"], 2))


def main():
    """Demonstrate all Session 5 invocation styles for beginners.

    Compared to Session 4 main(), this function now runs four scenarios to
    teach how one pipeline can be controlled via arguments:
    1) default call (no overrides),
    2) positional override call,
    3) keyword override call,
    4) fit-based call (use_fit=True).
    """

    make_print_status("Run 1: default arguments")
    _, _, default_result = setup_application()
    print_summary("Default Run", default_result)

    make_print_status("Run 2: positional argument overrides")
    _, _, positional_result = setup_application(
        "session5/iris_data.csv", 1.8, False)
    print_summary("Positional Run (threshold=1.8)", positional_result)

    make_print_status("Run 3: keyword argument overrides")
    _, _, keyword_result = setup_application(threshold=2.2, print_each=False)
    print_summary("Keyword Run (threshold=2.2)", keyword_result)

    make_print_status("Run 4: fit-based threshold")
    _, _, fit_result = setup_application(print_each=False, use_fit=True)
    print_summary("Fit-Based Run (use_fit=True)", fit_result)


if __name__ == "__main__":
    main()
