"""session 7 student template: Inheritance (Refactoring S6 into S7)."""

# session 7 starter dataset
dataset = [
    {"id": "flower1", "petal_length": 1.4, "species": "setosa"},
    {"id": "flower2", "petal_length": 1.5, "species": "setosa"},
    {"id": "flower3", "petal_length": 1.3, "species": "setosa"},
    {"id": "flower4", "petal_length": 4.5, "species": "versicolor"},
    {"id": "flower5", "petal_length": 4.7, "species": "versicolor"},
    {"id": "flower6", "petal_length": 5.1, "species": "virginica"},
    {"id": "flower7", "petal_length": 6.0, "species": "virginica"},
]


class ClassifierBase:
    """Base class holding shared logic (evaluation loop)."""

    def __init__(self):
        # We don't store threshold here anymore, because not all classifiers use a threshold!
        pass

    def predict(self, sample):
        # TODO 1: Raise NotImplementedError to force child classes to write their own
        raise NotImplementedError("Child class must implement predict()")

    def evaluate(self, dataset):
        # TODO 2: Reuse the evaluate logic from session 6
        # Key change: it calls self.predict(), which the child provides
        correct = 0
        total = 0

        # for sample in dataset:
        #    ...

        return 0.0


class RuleClassifier(ClassifierBase):
    """Child class: session 6 logic wrapped in inheritance."""

    def __init__(self, threshold=2.0):
        super().__init__()  # Good practice to initialize parent
        self.threshold = threshold

    def predict(self, sample):
        # TODO 3: Implement the familiar rule logic
        return "setosa"


class NearestCentroidClassifier(ClassifierBase):
    """Child class: A new way to classify (by distance to a target value)."""

    def __init__(self, setosa_target=1.4, non_setosa_target=4.5):
        super().__init__()
        self.setosa_target = setosa_target
        self.non_setosa_target = non_setosa_target

    def predict(self, sample):
        # TODO 4: Calculate distance to both targets
        # val = sample["petal_length"]
        # dist_setosa = abs(val - self.setosa_target)
        # dist_non = abs(val - self.non_setosa_target)

        # Return "setosa" if closer to setosa_target, else "not_setosa"
        return "setosa"


def main():
    print("=== Student Label ===")
    student_id = "YOUR_ID_HERE"
    full_name = "YOUR_FULL_NAME_HERE"
    print("Student ID:", student_id)
    print("Full Name:", full_name)

    # TODO 5: Create objects for both types
    rule_clf = RuleClassifier(threshold=2.0)
    nn_clf = NearestCentroidClassifier()

    # TODO 6: Evaluate them
    # Note: Both use the same .evaluate() method inherited from ClassifierBase!
    print("Rule Classifier Acc:", rule_clf.evaluate(dataset))
    print("Nearest Centroid Acc:", nn_clf.evaluate(dataset))


if __name__ == "__main__":
    main()
