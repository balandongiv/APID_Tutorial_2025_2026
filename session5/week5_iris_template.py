"""session 5 student template: extend session 4 functions with positional and keyword args.

This template keeps Session 4 core classifier logic unchanged and adds
Session 5 interface features: optional arguments, positional/keyword calling
styles, and a fit-based threshold path imported from another module.
"""

import csv
try:
    from session5.helper.iris_utils import fit_threshold_from_setosa
except ModuleNotFoundError:
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
    """Load Iris data with optional filepath fallback.

    Session 4 required filepath as a mandatory positional argument.
    Session 5 changes filepath to optional (default None), then applies a
    default path when no value is given.
    """
    # TODO 1: Add default filepath handling
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
    """Run prediction loop with optional per-sample logging.

    Session 4 always printed each sample line.
    Session 5 adds print_each=True so callers can switch logging on or off.
    """
    metrics = {"correct": 0, "wrong": 0, "total": 0, "y_pred_list": []}

    for sample in dataset:
        y_pred = classify_sample(sample, settings)
        y_true = determine_true_binary_label(sample, settings)
        update_metrics(metrics, y_pred, y_true)

        # TODO 2: Use print_each to control output
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
    """Load, configure, and execute experiment pipeline with Session 5 controls.

    Compared to Session 4, this function now supports optional parameter
    overrides and a fit-based threshold path.
    """
    dataset = load_iris_data(filepath)
    settings = get_default_settings()

    # TODO 3: Apply fit or threshold overrides
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
    """Entry point for session 5 runs."""
    print("=== Student Label ===")
    student_id = "YOUR_ID_HERE"
    full_name = "YOUR_FULL_NAME_HERE"
    print("Student ID:", student_id)
    print("Full Name:", full_name)

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
