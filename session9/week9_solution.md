# session 9 Solution (Module Convener)

Reference answer:

- `session9/session9_answer.py`

Supporting files:

- `session9/iris_eval.py`
- `session9/session9_plot_accuracy.py`
- `session8/iris_session8_utils.py`
- `session8/iris_session8_models.py`

## What was completed

1. Reused Session 8 modules for arrays, centroids, and model interface.
2. Implemented parameterized `run_experiment(test_size, seed, metric)`.
3. Implemented repeated-trial experiment runner returning list-of-dicts.
4. Saved results to CSV and Excel (Excel is optional/fallback-safe).
5. Generated plots derived from Session 8 products:
   - scatter with centroid overlay,
   - box plot of accuracy by metric,
   - line plot of mean accuracy vs test size.

## Expected behavior

- Prints trial-by-trial accuracy logs.
- Saves:
  - `session9/iris_experiment_results.csv`
  - `session9/iris_experiment_results.xlsx` (if writer available)
  - `session9/session9_scatter_centroids.png`
  - `session9/session9_accuracy_boxplot.png`
  - `session9/session9_accuracy_lineplot.png`

## Quick run

```bash
python session9/session9_answer.py
```
