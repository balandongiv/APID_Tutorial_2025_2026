# session 7 Solution (Module Convener)

Reference answer:

- `session7/session7_answer.py`

## What was completed

1. Implemented a shared `ClassifierBase` with `predict_all()` and `evaluate()`.
2. Implemented child classes that override `predict_one(...)`:
   - `RuleSetosaClassifier`
   - `RuleVirginicaClassifier`
   - `NearestCentroidClassifier`
3. Used `super()` in child constructors and `fit()` methods.
4. Evaluated all models using one shared evaluation pipeline for fair comparison.

## Expected behavior

- Each model prints `Correct`, `Wrong`, `Total`, and `Accuracy (%)`.
- All models are trained/evaluated in a single loop.
- Inheritance relationships are explicit and readable.

## Quick run

```bash
python session7/session7_answer.py
```
