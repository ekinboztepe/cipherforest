# Changelog

## 0.3.0

- Categorical features: CatBoost models trained with `cat_features` (one-hot and CTR
  splits, `max_ctr_complexity=1`) are served exactly. The client sends categories as
  they are and computes each split's bit itself (see SECURITY.md for what that reveals).
- Missing values: NaN routes as the model's `nan_mode` (Min or Max) says.
- Multiclass models (`MultiClassModel`): one encrypted pass returns every class's score
  and SHAP values, the comparisons shared across classes (about 2.2 times faster than a
  model per class on Covertype's 7 classes). Regression models are tested too.
- Retraining without new keys: `compile_model(..., same_keys_as=old client.json)` and
  `Client.load(..., keys=old.secret_keys())`. A model that would need other keys is
  refused instead of crashing the server.
- Outputs are 12 bits by default instead of 8: the served probability of a 30-tree
  model stays within 0.005 of CatBoost's instead of up to 0.11, at no cost for the
  SHAP output. Pass `output_bits=8` for faster score and decision circuits.
- `BudgetedServer(..., state=path)` keeps request counts across restarts, and counts
  under a lock so concurrent requests cannot overrun a budget.
- `KeyStore`: clients' keys on disk, only the most recently used few in memory
  (about 2 GB each once used).
- Command line: `cipherforest prepare | explain | compile`.
- Examples: an HTTP server with a request budget per key, and a distilled copy of a
  scikit-learn model with teacher-aligned explanations.
- Releases are published to PyPI from version tags.
- Measured: CatBoost against Concrete-ML on one machine, copying attacks on the
  30-tree model, categorical features, key reuse and output precision
  (experiments/RESULTS.md).

## 0.2.0

- Public API (`from_catboost`, `distill`, `explain`, `compile_model`, `export`,
  `Client`, `Server`, `BudgetedServer`), score and decision outputs, teacher-aligned
  explanation tables, Concrete-ML's error settings by default, SECURITY.md and a
  deployment guide.
