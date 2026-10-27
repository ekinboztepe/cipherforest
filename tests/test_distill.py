import numpy as np

from cipherforest.distill import distill_ensemble, fit_oblivious_tree
from cipherforest.quantize import FeatureQuantizer


def test_leaf_values_are_per_leaf_means() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 4))
    y = rng.normal(size=500)
    tree = fit_oblivious_tree(x, y, depth=3, thresholds=FeatureQuantizer.fit(x, 16).edges)
    leaves = tree.leaf_index(x)
    for leaf in np.unique(leaves):
        np.testing.assert_allclose(tree.leaf_values[leaf], y[leaves == leaf].mean())


def test_training_error_does_not_increase_with_depth() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 5))
    y = np.sin(x[:, 0]) + (x[:, 1] > 0.3) - 0.5 * x[:, 2] * x[:, 3]
    edges = FeatureQuantizer.fit(x, 32).edges
    errors = [np.mean((fit_oblivious_tree(x, y, d, edges).predict(x) - y) ** 2) for d in range(1, 7)]
    assert all(b <= a + 1e-12 for a, b in zip(errors, errors[1:], strict=False))


def test_recovers_an_axis_aligned_step_function_exactly() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(size=(1000, 3))
    edges = FeatureQuantizer.fit(x, 32).edges
    y = 3.0 * (x[:, 0] > edges[0][10]) - 2.0 * (x[:, 2] > edges[2][20])
    tree = fit_oblivious_tree(x, y, depth=2, thresholds=edges)
    np.testing.assert_allclose(tree.predict(x), y, atol=1e-12)


def test_boosting_reduces_training_error() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(800, 6))
    y = np.tanh(x[:, 0] * x[:, 1]) + 0.3 * x[:, 4]
    edges = FeatureQuantizer.fit(x, 32).edges
    errors = [np.mean((distill_ensemble(x, y, 4, t, edges).predict(x) - y) ** 2) for t in (1, 2, 3)]
    assert errors[0] > errors[1] > errors[2]
