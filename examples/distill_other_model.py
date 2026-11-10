"""Serve a model the circuit cannot take directly (here scikit-learn's gradient boosting).

    pip install "cipherforest[fhe]" scikit-learn shap
    python examples/distill_other_model.py

XGBoost, LightGBM and forests go the same way: fit a small oblivious copy to the
original's scores, then serve the copy. The copy is what the circuit explains, and
exactly; it agrees with the original only as well as its size allows, so check
both numbers below before deploying. With teacher_shap the explanation tables are
fitted to the original's SHAP values instead: the prediction stays the copy's,
and the explanation follows the original more closely, but it is then no longer
the copy's own Shapley values. Steps 1 to 3 run on any OS.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import cipherforest


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))


# 1. The original model, and its SHAP values against a fixed background.
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0, stratify=y)
teacher = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0).fit(x_train, y_train)
background = x_train[np.random.default_rng(0).choice(len(x_train), size=100, replace=False)]
explainer = shap.TreeExplainer(teacher, data=background, feature_perturbation="interventional")
fit_rows = x_train[:300]

# 2. Copies fitted to its log-odds: 10 trees of depth 4, plain and teacher-aligned.
target = teacher.decision_function(x_train)
plain = cipherforest.distill(x_train, target, depth=4, n_trees=10, background=background)
aligned = cipherforest.distill(x_train, target, depth=4, n_trees=10, background=background,
                               teacher_shap=(fit_rows, explainer.shap_values(fit_rows)))

# 3. How close each copy is to the original, on rows neither saw.
teacher_shap = explainer.shap_values(x_test)
print(f"AUC: original {roc_auc_score(y_test, teacher.decision_function(x_test)):.4f}", end="")
for name, model in (("copy", plain), ("aligned copy", aligned)):
    score, values = cipherforest.explain(model, x_test)
    print(f", {name} {roc_auc_score(y_test, score):.4f}", end="")
print()
for name, model in (("copy", plain), ("aligned copy", aligned)):
    score, values = cipherforest.explain(model, x_test)
    print(f"{name}: SHAP cosine to the original {cosine(values, teacher_shap):.3f}, "
          f"score correlation {np.corrcoef(score, teacher.decision_function(x_test))[0, 1]:.3f}")

try:
    from concrete import fhe  # noqa: F401
except ImportError as error:
    raise SystemExit(f"stopping before the encrypted part, concrete-python does not load: {error}") from error

# 4. Serve the aligned copy encrypted; the client gets what step 3 printed.
with TemporaryDirectory() as tmp:
    out = Path(tmp)
    cipherforest.export(cipherforest.compile_model(aligned), aligned, out)
    server = cipherforest.Server(out / "server.zip")
    client = cipherforest.Client.load(out / "client.json")
    keys = server.load_keys(client.evaluation_keys())
    score, values = client.decrypt(server.run(client.encrypt(x_test[0]), keys))
    want_score, want_values = cipherforest.explain(aligned, x_test[:1])
    assert abs(score - want_score[0]) < 1e-9 and np.allclose(values, want_values[0])
    print("encrypted run agrees with the clear one:", round(score, 3))
