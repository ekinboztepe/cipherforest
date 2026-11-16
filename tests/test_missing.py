"""Missing values: NaN routes like CatBoost's nan_mode, in the clear and after a round trip."""

import numpy as np
import pytest

import cipherforest
from cipherforest.quantize import FeatureQuantizer, QuantizedModel


def data(seed: int, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32).astype(np.float64)
    y = (x[:, 0] + 0.5 * x[:, 1] - x[:, 2] * x[:, 3] > 0).astype(int)
    x[rng.random(n) < 0.25, 0] = np.nan
    x[rng.random(n) < 0.1, 2] = np.nan
    return x, y


@pytest.mark.parametrize("mode", ["Min", "Max"])
def test_catboost_with_nans_matches_its_raw_score(mode: str) -> None:
    catboost = pytest.importorskip("catboost")
    x, y = data(0)
    cb = catboost.CatBoostClassifier(iterations=8, depth=3, verbose=0, random_seed=0, nan_mode=mode,
                                     allow_writing_files=False).fit(x, y)
    model = cipherforest.from_catboost(cb, background=x[:60], x_train=x)
    assert model.quantizer.nan_high[:1] == ((True,) if mode == "Max" else ())[:1]
    pred, shap = cipherforest.explain(model, x)
    np.testing.assert_allclose(pred, cb.predict(x, prediction_type="RawFormulaVal"), atol=len(model.trees) / model.scale)
    np.testing.assert_allclose(pred, model.base_value + shap.sum(axis=1), atol=1e-9)
    # a missing value lands in the same leaves as the most extreme value on its side
    extreme = x.copy()
    extreme[np.isnan(x)] = 1e9 if mode == "Max" else -1e9
    np.testing.assert_array_equal(cipherforest.explain(model, extreme)[1], shap)


def test_nan_side_survives_serialisation() -> None:
    quantizer = FeatureQuantizer((np.array([0.0, 1.0]), np.array([2.0])), (False, True))
    codes = quantizer.transform(np.array([[np.nan, np.nan], [0.5, 3.0]]))
    np.testing.assert_array_equal(codes, [[0, 1], [1, 1]])
    again = FeatureQuantizer.from_doc(quantizer.to_doc())
    assert again.nan_high == (False, True)
    np.testing.assert_array_equal(again.transform(np.array([[np.nan, np.nan]])), [[0, 1]])


def test_quantiles_and_distillation_ignore_nans() -> None:
    x, y = data(1)
    quantizer = FeatureQuantizer.fit(x, 8)
    assert all(np.isfinite(e).all() and e.size for e in quantizer.edges)
    model = cipherforest.distill(x, y, depth=3, n_trees=2, n_bins=8, background=40)
    again = QuantizedModel.from_json(model.to_json())
    np.testing.assert_array_equal(cipherforest.explain(again, x)[1], cipherforest.explain(model, x)[1])


@pytest.mark.fhe
def test_encrypted_run_with_missing_values(tmp_path) -> None:
    pytest.importorskip("concrete")
    from dataclasses import replace

    x, y = data(2)
    model = cipherforest.distill(x, y, depth=3, n_trees=2, n_bins=8, background=40)
    # feature 2's NaN above every edge, as nan_mode="Max" would put it
    model = replace(model, quantizer=replace(model.quantizer, nan_high=(False, False, True, False)))
    cipherforest.export(cipherforest.compile_model(model), model, tmp_path)
    server = cipherforest.Server(tmp_path / "server.zip")
    client = cipherforest.Client.load(tmp_path / "client.json")
    keys = server.load_keys(client.evaluation_keys())
    rows = x[np.isnan(x).any(axis=1)][:3]
    want_pred, want_shap = cipherforest.explain(model, rows)
    for row, p, s in zip(rows, want_pred, want_shap, strict=True):
        pred, shap = client.decrypt(server.run(client.encrypt(row), keys))
        assert abs(pred - p) < 1e-9 and np.allclose(shap, s)
