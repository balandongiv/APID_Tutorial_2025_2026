"""session 6 student template: classes and objects (Refactoring S5 into S6)."""

# session 6 starter dataset (Iris-only)
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
        # TODO 1: Store threshold as an object attribute
        self.threshold = threshold
        pass

    def predict(self, sample):
        """Predict 'setosa' or 'not_setosa' for a single sample."""
        # TODO 2: Re-implement the session 5 'detect_setosa' logic here
        # Remember: use self.threshold instead of settings['threshold']
        return "setosa"

    def evaluate(self, dataset):
        """Evaluate the classifier on a full dataset and return accuracy (0.0 to 1.0)."""
        # TODO 3: Implement the evaluation loop
        # Count correct predictions vs total samples
        correct = 0
        total = 0

        # for sample in dataset:
        #    ...

        return 0.0


def main():
    print("=== Student Label ===")
    student_id = "YOUR_ID_HERE"
    full_name = "YOUR_FULL_NAME_HERE"
    print("Student ID:", student_id)
    print("Full Name:", full_name)

    # TODO 4: Create a classifier object
    my_clf = IrisRuleClassifier()  # Default threshold 2.0
    print("Created Classifier:", my_clf.threshold)

    # TODO 5: Evaluate it
    accuracy = my_clf.evaluate(dataset)
    print("Accuracy (Default):", accuracy)

    # TODO 6: Create customized classifiers and compare
    clf_strict = IrisRuleClassifier(threshold=1.5)
    clf_loose = IrisRuleClassifier(threshold=2.5)

    print("Accuracy (Strict 1.5):", clf_strict.evaluate(dataset))
    print("Accuracy (Loose 2.5):", clf_loose.evaluate(dataset))


if __name__ == "__main__":
    main()
