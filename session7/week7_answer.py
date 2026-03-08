"""session 7 Answer: Inheritance and Polymorphism."""

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
        # The parent doesn't know how to predict!
        # Child classes MUST replace this method with their own logic.
        raise NotImplementedError("Child class must implement predict()")

    def evaluate(self, dataset):
        # This logic is exactly the same as session 6!
        # It calls self.predict(), which will be provided by the child class.
        correct = 0
        total = 0

        for sample in dataset:
            prediction = self.predict(sample)

            # Simple binary truth check
            truth = "setosa" if sample["species"] == "setosa" else "not_setosa"

            if prediction == truth:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0


class RuleClassifier(ClassifierBase):
    """Child class: session 6 logic wrapped in inheritance."""

    def __init__(self, threshold=2.0):
        super().__init__()  # Initialize the parent
        self.threshold = threshold

    def predict(self, sample):
        # Specific logic for THIS classifier
        if sample["petal_length"] < self.threshold:
            return "setosa"
        return "not_setosa"


class NearestCentroidClassifier(ClassifierBase):
    """Child class: A new way to classify (by distance to a target value)."""

    def __init__(self, setosa_target=1.4, non_setosa_target=4.5):
        super().__init__()
        self.setosa_target = setosa_target     # Target petal_length for setosa
        self.non_setosa_target = non_setosa_target  # Target for others

    def predict(self, sample):
        # Logic: Which target value is the sample's petal_length closer to?
        val = sample["petal_length"]
        dist_to_setosa = abs(val - self.setosa_target)
        dist_to_non = abs(val - self.non_setosa_target)

        if dist_to_setosa < dist_to_non:
            return "setosa"
        return "not_setosa"


def main():
    print("=== Student Label ===")
    student_id = "MODEL_ANSWER"
    full_name = "MODULE_CONVENER"
    print("Student ID:", student_id)
    print("Full Name:", full_name)

    # 1. Create objects for both types
    rule_clf = RuleClassifier(threshold=2.0)
    nn_clf = NearestCentroidClassifier()

    # 2. Evaluate them
    # Note: Both use the same .evaluate() method inherited from ClassifierBase!
    print("Rule Classifier Accuracy:", rule_clf.evaluate(dataset))
    print("Nearest Centroid Accuracy:", nn_clf.evaluate(dataset))


if __name__ == "__main__":
    main()
