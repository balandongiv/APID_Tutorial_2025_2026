"""session 6 Answer: Basic Classes and Objects."""

# session 6 starter dataset (Iris-only for simplicity)
dataset = [
    {"id": "flower1", "petal_length": 1.4, "species": "setosa"},
    {"id": "flower2", "petal_length": 1.5, "species": "setosa"},
    {"id": "flower3", "petal_length": 1.3, "species": "setosa"},
    {"id": "flower4", "petal_length": 4.5, "species": "versicolor"},
    {"id": "flower5", "petal_length": 4.7, "species": "versicolor"},
    {"id": "flower6", "petal_length": 5.1, "species": "virginica"},
    {"id": "flower7", "petal_length": 6.0, "species": "virginica"},
]


class IrisRuleClassifier:
    """A class version of our session 5 function-based classifier."""

    def __init__(self, threshold=2.0):
        """Constructor: Sets up the object's data (attributes)."""
        # Store threshold as an instance variable
        self.threshold = threshold

        # We can also store labels here instead of hardcoding
        self.positive_label = "setosa"
        self.negative_label = "not_setosa"

    def predict(self, sample):
        """Predict 'setosa' or 'not_setosa' for a single sample."""
        # Logic remains the same as S5, but uses 'self.threshold'
        if sample["petal_length"] < self.threshold:
            return self.positive_label
        return self.negative_label

    def evaluate(self, dataset):
        """Evaluate the classifier on a full dataset and return accuracy (0.0 to 1.0)."""
        correct = 0
        total = 0

        for sample in dataset:
            # 1. Ask the object to predict
            prediction = self.predict(sample)

            # 2. Determine truth (simplified binary check for session 6)
            truth = "setosa" if sample["species"] == "setosa" else "not_setosa"

            # 3. Score
            if prediction == truth:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy


def main():
    print("=== Student Label ===")
    student_id = "MODEL_ANSWER"
    full_name = "MODULE_CONVENER"
    print("Student ID:", student_id)
    print("Full Name:", full_name)

    # 1. Create a default classifier object (threshold=2.0)
    clf_default = IrisRuleClassifier()
    print("\n--- Default Classifier (2.0) ---")
    print("Threshold:", clf_default.threshold)
    acc_default = clf_default.evaluate(dataset)
    print(f"Accuracy: {acc_default:.2f}")

    # 2. Create customized classifiers
    clf_strict = IrisRuleClassifier(threshold=1.5)
    clf_loose = IrisRuleClassifier(threshold=2.5)

    print("\n--- Strict Classifier (1.5) ---")
    acc_strict = clf_strict.evaluate(dataset)
    print(f"Accuracy: {acc_strict:.2f}")

    print("\n--- Loose Classifier (2.5) ---")
    acc_loose = clf_loose.evaluate(dataset)
    print(f"Accuracy: {acc_loose:.2f}")


if __name__ == "__main__":
    main()
