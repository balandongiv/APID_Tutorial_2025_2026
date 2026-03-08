"""session 5 decomposition template: break classify logic into smaller functions."""


def determine_binary_label(sample, settings):
    """Predict setosa/not_setosa based on configured threshold."""
    # TODO: implement threshold logic
    if sample[settings["feature_name"]] < settings["threshold"]:
        return settings["positive_label"]
    return settings["negative_label"]


def evaluate_prediction(y_pred, sample, settings):
    """Return y_true and correctness flag."""
    # TODO: derive true binary label from species
    if sample[settings["label_key"]] == settings["positive_label"]:
        y_true = settings["positive_label"]
    else:
        y_true = settings["negative_label"]

    return y_true, (y_pred == y_true)


def classify_sample(sample, settings):
    """Wrapper that uses decomposed helper(s)."""
    # TODO: call determine_binary_label
    return determine_binary_label(sample, settings)
