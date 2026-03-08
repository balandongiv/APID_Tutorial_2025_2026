# session 6 Solution (Module Convener)

Reference answer:

- `session6/session6_answer.py`

## What was completed

1. Implemented `IrisRuleClassifier` with `__init__`, `__str__`, `predict_one`, `predict_all`, `fit`, and `evaluate`.
2. Kept session 4/5 rule behavior stable: `petal_length < threshold -> setosa` else `not_setosa`.
3. Created multiple classifier objects and evaluated them in a loop.
4. Included student label output block.

## Expected behavior

- Prints readable classifier objects.
- Prints example predictions for short/long petal lengths.
- Shows threshold update before/after `fit()`.
- Shows looped metrics (`Correct`, `Wrong`, `Total`, `Accuracy (%)`) for multiple thresholds.

## Quick run

```bash
python session6/session6_answer.py
```
