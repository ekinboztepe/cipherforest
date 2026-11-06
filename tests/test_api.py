"""The short public path, in the clear."""

from pathlib import Path

import numpy as np
import pytest

import cipherforest
from cipherforest.shap_table import build_shap_table, explain_ensemble


def data(seed: int, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5)).astype(np.float32).astype(np.float64)
    y = (x[:, 0] + x[:, 1] * x[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return x, y


def test_distill_explains_itself_and_sums_to_the_prediction() -> None:
    x, y = data(0)
    model = cipherforest.distill(x, y, depth=4, n_trees=2, n_bins=16, background=50)
    pred, shap = cipherforest.explain(model, x[:40])
    np.testing.assert_allclose(pred, model.base_value + shap.sum(axis=1), rtol=0, atol=1e-12)
    assert np.corrcoef(pred, y[:40])[0, 1] > 0.5


def test_catboost_model_object_and_json_text_give_the_same_model(tmp_path: Path) -> None:
    catboost = pytest.importorskip("catboost")
    x, y = data(1)
    cb = catboost.CatBoostClassifier(iterations=5, depth=3, verbose=0, random_seed=0, allow_writing_files=False).fit(x, y)
    path = tmp_path / "m.json"
    cb.save_model(str(path), format="json")
    a = cipherforest.from_catboost(cb, background=x[:50], x_train=x)
    b = cipherforest.from_catboost(path.read_text(), background=x[:50], x_train=x)
    assert a.to_json() == b.to_json()
    pred, shap = cipherforest.explain(a, x[:100])
    # raw log-odds, up to the fixed-point rounding of the outputs
    np.testing.assert_allclose(pred, cb.predict(x[:100], prediction_type="RawFormulaVal"), atol=len(a.trees) / a.scale)
    # padding edges: client.json lists more values than the model splits on
    assert sum(e.size for e in a.quantizer.edges) > sum(t.features.size for t in a.trees)


def test_from_ensemble_matches_float_shap() -> None:
    x, y = data(2)
    model = cipherforest.distill(x, y, depth=3, n_trees=2, n_bins=16, background=x[:30])
    from cipherforest.distill import distill_ensemble

    ensemble = distill_ensemble(x, y.astype(float), 3, 2, model.quantizer.edges)
    tables = [build_shap_table(t, x[:30]) for t in ensemble.trees]
    _, shap = cipherforest.explain(model, x[:50])
    exact = explain_ensemble(ensemble, tables, x[:50], x.shape[1])
    assert np.abs(shap - exact).max() <= len(ensemble.trees) / model.scale


def test_fhe_names_load_lazily() -> None:
    assert "compile_model" in cipherforest.__all__
    with pytest.raises(AttributeError):
        _ = cipherforest.no_such_name


def test_distill_with_teacher_shap_keeps_the_prediction() -> None:
    x, y = data(4)
    plain = cipherforest.distill(x, y, depth=4, n_trees=2, n_bins=16, background=x[:40])
    _, own = cipherforest.explain(plain, x[:200])
    target = np.roll(own, 1, axis=1)  # any other division of the same totals
    aligned = cipherforest.distill(x, y, depth=4, n_trees=2, n_bins=16, background=x[:40], teacher_shap=(x[:200], target))
    p1, _ = cipherforest.explain(plain, x[:200])
    p2, s2 = cipherforest.explain(aligned, x[:200])
    tol = 2 * (len(plain.trees) / plain.scale + len(aligned.trees) / aligned.scale)
    np.testing.assert_allclose(p1, p2, atol=tol)
    assert np.mean((s2 - target) ** 2) < np.mean((own - target) ** 2)
