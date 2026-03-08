# session 8 Solution (Module Convener)

Reference answer:

- `session8/session8_answer.py`

Shared modules:

- `session8/iris_session8_utils.py`
- `session8/iris_session8_models.py`

## What was completed

1. Loaded Iris data into NumPy arrays (`X`, `y`) with CSV-first and built-in fallback behavior.
2. Demonstrated indexing/slicing and boolean masks.
3. Computed class centroids with `np.mean(..., axis=0)`.
4. Reused session 7 classifier interface through `NumpyCentroidClassifier` (`fit`, `predict_one`, `predict_all`, `evaluate`).
5. Compared `l2` and `l1` metrics.

## Expected behavior

- Prints array shapes/dtypes.
- Prints class counts and rounded centroids.
- Prints evaluation metrics and metric comparison accuracies.

## Quick run

```bash
python session8/session8_answer.py
```
