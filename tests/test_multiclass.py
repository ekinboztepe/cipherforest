"""Multiclass and regression CatBoost models."""

import numpy as np
import pytest

import cipherforest

catboost = pytest.importorskip("catboost")


def data(seed: int, n: int = 900) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32).astype(np.float64)
    y = (x[:, 0] > 0).astype(int) + (x[:, 1] * x[:, 2] > 0.3).astype(int)
    return x, y


def test_multiclass_scores_and_shap_match_catboost() -> None:
    x, y = data(0)
    cb = catboost.CatBoostClassifier(iterations=8, depth=3, verbose=0, random_seed=0, loss_function="MultiClass",
                                     allow_writing_files=False).fit(x, y)
    model = cipherforest.from_catboost(cb, background=x[:60], x_train=x)
    assert isinstance(model, cipherforest.MultiClassModel) and len(model.classes) == 3
    scores, shap = cipherforest.explain(model, x[:200])
    assert scores.shape == (200, 3) and shap.shape == (200, 3, 4)
    bound = max(len(m.trees) / (2 * m.scale) for m in model.classes)
    np.testing.assert_allclose(scores, cb.predict(x[:200], prediction_type="RawFormulaVal"), atol=bound)
    bases = np.array([m.base_value for m in model.classes])
    np.testing.assert_allclose(scores, bases + shap.sum(axis=2), atol=1e-9)
    assert np.mean(scores.argmax(axis=1) == cb.predict(x[:200]).ravel()) > 0.99

    again = cipherforest.model_from_json(model.to_json())
    np.testing.assert_array_equal(cipherforest.explain(again, x[:50])[1], shap[:50])


def test_regression_matches_catboost() -> None:
    x, _ = data(1)
    target = x[:, 0] * 3 + np.sin(x[:, 1])
    cb = catboost.CatBoostRegressor(iterations=10, depth=3, verbose=0, random_seed=0,
                                    allow_writing_files=False).fit(x, target)
    model = cipherforest.from_catboost(cb, background=x[:60], x_train=x)
    pred, _ = cipherforest.explain(model, x)
    np.testing.assert_allclose(pred, cb.predict(x), atol=len(model.trees) / (2 * model.scale))


@pytest.mark.fhe
@pytest.mark.parametrize(("output", "public"), [("shap", False), ("score", False), ("shap", True)])
def test_encrypted_multiclass(tmp_path, output: str, public: bool) -> None:
    pytest.importorskip("concrete")
    x, y = data(2, n=400)
    cb = catboost.CatBoostClassifier(iterations=3, depth=3, verbose=0, random_seed=0, loss_function="MultiClass",
                                     allow_writing_files=False).fit(x, y)
    model = cipherforest.from_catboost(cb, background=x[:40], x_train=x)
    cipherforest.export(cipherforest.compile_model(model, output=output, public_splits=public), model, tmp_path)
    server = cipherforest.Server(tmp_path / "server.zip")
    client = cipherforest.Client.load(tmp_path / "client.json")
    keys = server.load_keys(client.evaluation_keys())
    want_scores, want_shap = cipherforest.explain(model, x[:3])
    for row, s, phi in zip(x[:3], want_scores, want_shap, strict=True):
        scores, shap = client.decrypt(server.run(client.encrypt(row), keys))
        np.testing.assert_allclose(scores, s, atol=1e-9)
        if output == "shap":
            np.testing.assert_allclose(shap, phi, atol=1e-9)
        else:
            assert shap is None
