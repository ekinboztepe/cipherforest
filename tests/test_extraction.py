import numpy as np
import pytest

from cipherforest.distill import distill_ensemble
from cipherforest.extraction import CountingOracle, code_values, find_thresholds, plaintext_oracle, true_thresholds
from cipherforest.quantize import FeatureQuantizer, QuantizedModel, quantize_model
from cipherforest.shap_table import build_shap_table


def small_model(seed: int, depth: int, n_trees: int, n_features: int = 6) -> QuantizedModel:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(500, n_features))
    y = np.tanh(x[:, 0] * x[:, 1]) + (x[:, 2] > 0.3) + 0.5 * x[:, 3] + 0.1 * rng.normal(size=500)
    quantizer = FeatureQuantizer.fit(x, n_bins=16)
    ensemble = distill_ensemble(x, y, depth, n_trees, quantizer.edges)
    tables = [build_shap_table(tree, x[:30]) for tree in ensemble.trees]
    return quantize_model(ensemble, tables, quantizer, output_bits=8)


def test_code_values_hit_every_code() -> None:
    quantizer = FeatureQuantizer((np.array([-1.0, 0.0, 2.5]), np.array([]), np.array([7.0])))
    for f, edges in enumerate(quantizer.edges):
        x = np.zeros((code_values(edges).size, quantizer.n_features))
        x[:, f] = code_values(edges)
        np.testing.assert_array_equal(quantizer.transform(x)[:, f], np.arange(edges.size + 1))


@pytest.mark.parametrize(("depth", "n_trees"), [(2, 1), (4, 2), (6, 3)])
def test_finds_only_true_thresholds(depth: int, n_trees: int) -> None:
    model = small_model(depth, depth, n_trees)
    truth = true_thresholds(model)
    for use_shap in (True, False):
        found = find_thresholds(plaintext_oracle(model), model.quantizer, 1, np.random.default_rng(0), use_shap)
        assert found <= truth


@pytest.mark.parametrize(("depth", "n_trees"), [(2, 1), (4, 2), (6, 3)])
def test_shap_finds_every_threshold(depth: int, n_trees: int) -> None:
    model = small_model(depth, depth, n_trees)
    found = find_thresholds(plaintext_oracle(model), model.quantizer, 3, np.random.default_rng(0))
    assert found == true_thresholds(model)


def test_query_count() -> None:
    model = small_model(1, 3, 2)
    oracle = CountingOracle(plaintext_oracle(model))
    find_thresholds(oracle, model.quantizer, 2, np.random.default_rng(0))
    sweeps = sum(e.size + 1 for e in model.quantizer.edges if e.size)
    assert oracle.queries == 2 * sweeps
