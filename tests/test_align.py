import numpy as np

from cipherforest.align import align_tables
from cipherforest.distill import distill_ensemble
from cipherforest.quantize import FeatureQuantizer, quantize_model
from cipherforest.shap_table import build_shap_table, explain_ensemble


def setup(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(800, 6))
    y = np.tanh(x[:, 0] * x[:, 1]) + (x[:, 2] > 0.3) + 0.5 * x[:, 3] + 0.3 * x[:, 4] + 0.1 * rng.normal(size=800)
    quantizer = FeatureQuantizer.fit(x, n_bins=16)
    ensemble = distill_ensemble(x, y, 4, 3, quantizer.edges)
    tables = [build_shap_table(tree, x[:40]) for tree in ensemble.trees]
    return x, quantizer, ensemble, tables


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)))


def test_own_shap_as_target_changes_nothing() -> None:
    x, _, ensemble, tables = setup()
    own = explain_ensemble(ensemble, tables, x, x.shape[1])
    aligned = align_tables(ensemble, tables, x, own)
    for a, b in zip(aligned, tables, strict=True):
        np.testing.assert_allclose(a.values, b.values, atol=1e-8)


def test_leaf_totals_and_predictions_are_kept() -> None:
    x, _, ensemble, tables = setup(1)
    rng = np.random.default_rng(2)
    target = rng.normal(size=x.shape)
    aligned = align_tables(ensemble, tables, x, target)
    for a, b in zip(aligned, tables, strict=True):
        np.testing.assert_allclose(a.values.sum(axis=1), b.values.sum(axis=1), atol=1e-8)
        np.testing.assert_array_equal(a.players, b.players)
    before = explain_ensemble(ensemble, tables, x, x.shape[1]).sum(axis=1)
    after = explain_ensemble(ensemble, aligned, x, x.shape[1]).sum(axis=1)
    np.testing.assert_allclose(after, before, atol=1e-8)


def test_moves_towards_the_target_and_quantises() -> None:
    x, quantizer, ensemble, tables = setup(3)
    own = explain_ensemble(ensemble, tables, x, x.shape[1])
    # a target that splits each row's total differently: same sums, other split
    target = own[:, ::-1] * 0.0 + own.sum(axis=1, keepdims=True) / x.shape[1]
    aligned = align_tables(ensemble, tables, x, target, ridge=1e-3)
    new = explain_ensemble(ensemble, aligned, x, x.shape[1])
    assert np.mean((new - target) ** 2) < np.mean((own - target) ** 2)
    model = quantize_model(ensemble, aligned, quantizer, output_bits=8)
    pred, _ = model.decode(model.simulate(quantizer.transform(x)))
    np.testing.assert_allclose(pred, ensemble.predict(x), atol=len(ensemble.trees) / model.scale)
