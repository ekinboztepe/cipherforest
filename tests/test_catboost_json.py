"""Imported CatBoost models against CatBoost itself and against the shap package."""

from math import factorial
from pathlib import Path

import numpy as np
import pytest

catboost = pytest.importorskip("catboost")
shap = pytest.importorskip("shap")

from cipherforest.catboost_json import load_catboost_json  # noqa: E402
from cipherforest.quantize import FeatureQuantizer, quantize_model  # noqa: E402
from cipherforest.shap_table import build_shap_table, explain_ensemble  # noqa: E402


def data(seed: int, n: int = 600, n_features: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """float32-exact values, which is what CatBoost compares."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_features)).astype(np.float32).astype(np.float64)
    x[:, 1] = np.round(x[:, 1])  # ties on borders
    y = (x[:, 0] + x[:, 1] * x[:, 2] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return x, y


def fit_and_load(tmp_path: Path, x: np.ndarray, y: np.ndarray, **params: object) -> tuple[object, str]:
    params = {"verbose": 0, "random_seed": 0, "allow_writing_files": False, **params}
    model = catboost.CatBoostClassifier(**params).fit(x, y)
    path = tmp_path / "model.json"
    model.save_model(str(path), format="json")
    return model, path.read_text()


def eval_rows(x: np.ndarray, quantizer: FeatureQuantizer, rng: np.random.Generator) -> np.ndarray:
    """Held-out rows plus rows sitting exactly on every border."""
    on_edges = np.column_stack([rng.choice(e, size=100) if e.size else np.zeros(100) for e in quantizer.edges])
    return np.vstack([x, on_edges])


@pytest.mark.parametrize(("depth", "n_trees"), [(1, 1), (3, 4), (6, 10)])
def test_predictions_match_catboost(tmp_path: Path, depth: int, n_trees: int) -> None:
    x, y = data(depth)
    model, text = fit_and_load(tmp_path, x, y, depth=depth, iterations=n_trees, border_count=32)
    ensemble, quantizer = load_catboost_json(text)
    rows = eval_rows(x, quantizer, np.random.default_rng(0))
    np.testing.assert_allclose(ensemble.predict(rows), model.predict(rows, prediction_type="RawFormulaVal"), atol=1e-9)


def brute_force_shap(model: object, x: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Interventional Shapley values with CatBoost's own predictions as the value function."""
    d = x.shape[1]
    masks = (np.arange(1 << d)[:, None] >> np.arange(d)) & 1
    sizes = masks.sum(axis=1)
    weights = np.array([factorial(s) * factorial(d - s - 1) / factorial(d) for s in range(d)])
    out = np.zeros_like(x)
    for r, row in enumerate(x):
        hybrid = np.where(masks[:, None, :] == 1, row, background[None, :, :]).reshape(-1, d)
        v = model.predict(hybrid, prediction_type="RawFormulaVal").reshape(1 << d, -1).mean(axis=1)
        for j in range(d):
            without = np.flatnonzero(masks[:, j] == 0)
            out[r, j] = np.sum(weights[sizes[without]] * (v[without | (1 << j)] - v[without]))
    return out


@pytest.mark.parametrize(("depth", "n_trees"), [(2, 3), (4, 5), (6, 4)])
def test_shap_is_exact(tmp_path: Path, depth: int, n_trees: int) -> None:
    x, y = data(10 + depth)
    model, text = fit_and_load(tmp_path, x, y, depth=depth, iterations=n_trees, border_count=16)
    ensemble, quantizer = load_catboost_json(text)
    background = x[:40]
    rows = eval_rows(x[40:240], quantizer, np.random.default_rng(1))[::8]

    tables = [build_shap_table(tree, background) for tree in ensemble.trees]
    ours = explain_ensemble(ensemble, tables, rows, x.shape[1])
    np.testing.assert_allclose(ours, brute_force_shap(model, rows, background), atol=1e-9)


@pytest.mark.parametrize(("depth", "n_trees"), [(2, 3), (4, 5), (6, 4)])
def test_shap_matches_shap_package(tmp_path: Path, depth: int, n_trees: int) -> None:
    x, y = data(10 + depth)
    model, text = fit_and_load(tmp_path, x, y, depth=depth, iterations=n_trees, border_count=16)
    ensemble, quantizer = load_catboost_json(text)
    background = x[:40]
    rows = eval_rows(x[40:240], quantizer, np.random.default_rng(1))

    tables = [build_shap_table(tree, background) for tree in ensemble.trees]
    ours = explain_ensemble(ensemble, tables, rows, x.shape[1])
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    theirs = explainer.shap_values(rows)

    # shap computes in float32 here: its own sum misses CatBoost's raw score by ~1e-7, ours by ~1e-15
    np.testing.assert_allclose(ours, theirs, atol=1e-6)
    expected = ensemble.bias + sum(t.expected_value for t in tables)
    np.testing.assert_allclose(expected, explainer.expected_value, atol=1e-6)
    raw = model.predict(rows, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(expected + ours.sum(axis=1), raw, atol=1e-12)


@pytest.mark.parametrize("pad", [False, True])
def test_integer_routing_matches_float_routing(tmp_path: Path, pad: bool) -> None:
    x, y = data(20)
    _, text = fit_and_load(tmp_path, x, y, depth=5, iterations=6, border_count=32)
    extra = FeatureQuantizer.fit(x, n_bins=32) if pad else None
    ensemble, quantizer = load_catboost_json(text, extra_edges=extra)
    tables = [build_shap_table(tree, x[:30]) for tree in ensemble.trees]
    model = quantize_model(ensemble, tables, quantizer, output_bits=8)

    rows = eval_rows(x, quantizer, np.random.default_rng(2))
    codes = quantizer.transform(rows)
    for tree, qtree in zip(ensemble.trees, model.trees, strict=True):
        np.testing.assert_array_equal(qtree.leaf_index(codes), tree.leaf_index(rows))


def test_padding_keeps_the_borders_and_adds_quantiles(tmp_path: Path) -> None:
    x, y = data(30)
    _, text = fit_and_load(tmp_path, x, y, depth=4, iterations=3, border_count=32)
    _, plain = load_catboost_json(text)
    extra = FeatureQuantizer.fit(x, n_bins=32)
    _, padded = load_catboost_json(text, extra_edges=extra)
    for b, e, p in zip(plain.edges, extra.edges, padded.edges, strict=True):
        assert np.all(np.isin(b, p)) and np.all(np.isin(e, p))


def test_rejects_what_it_cannot_serve_exactly(tmp_path: Path) -> None:
    x, y = data(40)
    _, text = fit_and_load(tmp_path, x, y, depth=3, iterations=2, grow_policy="Depthwise")
    with pytest.raises(ValueError, match="oblivious"):
        load_catboost_json(text)

    _, text = fit_and_load(tmp_path, x, np.arange(len(y)) % 3, depth=3, iterations=2, loss_function="MultiClass")
    with pytest.raises(ValueError, match="single-output"):
        load_catboost_json(text)

    with_nans = x.copy()
    with_nans[::7, 0] = np.nan
    _, text = fit_and_load(tmp_path, with_nans, y, depth=3, iterations=2)
    with pytest.raises(ValueError, match="NaN"):
        load_catboost_json(text)

    pool = catboost.Pool([[str(i % 3), *row] for i, row in enumerate(x.tolist())], y, cat_features=[0])
    model = catboost.CatBoostClassifier(depth=3, iterations=2, verbose=0, allow_writing_files=False).fit(pool)
    model.save_model(str(tmp_path / "cat.json"), format="json")
    with pytest.raises(ValueError, match="numeric"):
        load_catboost_json((tmp_path / "cat.json").read_text())
