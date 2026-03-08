"""session 5 additional function template for decomposition practice."""

import json


def load_iris_data(filepath):
    """Load JSON file and return parsed dictionary."""
    # TODO: complete loader
    with open(filepath, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def setup_experiment(data):
    """Return settings and dataset from loaded data."""
    # TODO: extract fields
    settings = data["settings"]
    dataset = data["dataset"]
    return settings, dataset


def initialize_predictions(dataset, settings, classify_fn):
    """Run prediction loop and return summary metrics."""
    correct = 0
    wrong = 0
    total = 0

    for sample in dataset:
        # TODO: classify one sample
        y_pred = classify_fn(sample, settings)

        if sample[settings["label_key"]] == settings["positive_label"]:
            y_true = settings["positive_label"]
        else:
            y_true = settings["negative_label"]

        if y_pred == y_true:
            correct += 1
        else:
            wrong += 1

        total += 1

    accuracy = (correct / total) * 100 if total else 0.0
    return {
        "correct": correct,
        "wrong": wrong,
        "total": total,
        "accuracy": accuracy,
    }


def setup_application(filepath, classify_fn):
    """High-level pipeline."""
    # TODO: integrate helper functions
    data = load_iris_data(filepath)
    settings, dataset = setup_experiment(data)
    result = initialize_predictions(dataset, settings, classify_fn)
    return settings, dataset, result
