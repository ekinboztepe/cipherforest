"""Categorical features: one-hot and CTR splits reproduce CatBoost exactly."""

import numpy as np
import pytest

import cipherforest
from cipherforest.quantize import QuantizedModel

catboost = pytest.importorskip("catboost")


def data(seed: int, n: int = 1500) -> list[list[object]]:
    rng = np.random.default_rng(seed)
    city = rng.choice(["ankara", "izmir", "bursa", "van", "rize", "mus"], n)
    plan = rng.choice([1, 2, 3], n)  # integer categories
    income = rng.normal(size=n).astype(np.float32).astype(np.float64)
    y = (np.isin(city, ["ankara", "rize"]) ^ (income > 0.5) ^ (plan == 2)).astype(int)
    rows = [[c, float(i), int(p)] for c, i, p in zip(city, income, plan)]
    return rows, y


@pytest.mark.parametrize("one_hot_max_size", [10, 1])  # one-hot splits, then CTR splits
def test_categorical_model_matches_catboost(one_hot_max_size: int) -> None:
    rows, y = data(0)
    cb = catboost.CatBoostClassifier(iterations=10, depth=3, verbose=0, random_seed=0, cat_features=[0, 2],
                                     one_hot_max_size=one_hot_max_size, max_ctr_complexity=1,
                                     allow_writing_files=False).fit(rows, y)
    model = cipherforest.from_catboost(cb, background=rows[:80], x_train=rows)
    assert model.quantizer.categories[0] == ("ankara", "bursa", "izmir", "mus", "rize", "van")
    test = rows[:300] + [["istanbul", 0.3, 4], ["", -1.0, 1]]  # unseen categories
    pred, shap = cipherforest.explain(model, test)
    raw = cb.predict(catboost.Pool(test, cat_features=[0, 2]), prediction_type="RawFormulaVal")
    np.testing.assert_allclose(pred, raw, atol=len(model.trees) / model.scale)
    np.testing.assert_allclose(pred, model.base_value + shap.sum(axis=1), atol=1e-9)
    assert np.abs(shap[:, 0]).max() > 0  # the city gets credit

    again = QuantizedModel.from_json(model.to_json())
    np.testing.assert_array_equal(cipherforest.explain(again, test)[1], shap)


def test_feature_combinations_are_rejected() -> None:
    rows, y = data(1)
    cb = catboost.CatBoostClassifier(iterations=10, depth=3, verbose=0, random_seed=0, cat_features=[0, 2],
                                     one_hot_max_size=1, max_ctr_complexity=2, allow_writing_files=False).fit(rows, y)
    with pytest.raises(ValueError, match="combination|several"):
        cipherforest.from_catboost(cb, background=rows[:80], x_train=rows)


@pytest.mark.fhe
def test_encrypted_run_with_categories(tmp_path) -> None:
    pytest.importorskip("concrete")
    rows, y = data(2, n=600)
    cb = catboost.CatBoostClassifier(iterations=4, depth=3, verbose=0, random_seed=0, cat_features=[0, 2],
                                     one_hot_max_size=1, max_ctr_complexity=1, allow_writing_files=False).fit(rows, y)
    model = cipherforest.from_catboost(cb, background=rows[:40], x_train=rows)
    cipherforest.export(cipherforest.compile_model(model), model, tmp_path)
    server = cipherforest.Server(tmp_path / "server.zip")
    client = cipherforest.Client.load(tmp_path / "client.json")
    keys = server.load_keys(client.evaluation_keys())
    for row in rows[:3] + [["istanbul", 0.1, 9]]:
        want_pred, want_shap = cipherforest.explain(model, [row])
        pred, shap = client.decrypt(server.run(client.encrypt(row), keys))
        assert abs(pred - want_pred[0]) < 1e-9 and np.allclose(shap, want_shap[0])
