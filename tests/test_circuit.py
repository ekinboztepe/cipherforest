"""The compiled circuit has to match the integer reference bit for bit. Needs concrete-python."""

import numpy as np
import pytest

pytest.importorskip("concrete.fhe")

from cipherforest.circuit import compile_model  # noqa: E402
from cipherforest.distill import distill_ensemble  # noqa: E402
from cipherforest.quantize import FeatureQuantizer, QuantizedModel, quantize_model  # noqa: E402
from cipherforest.shap_table import build_shap_table  # noqa: E402

pytestmark = pytest.mark.fhe


def small_model(seed: int, depth: int, n_trees: int, n_features: int = 5) -> QuantizedModel:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(400, n_features))
    y = np.tanh(x[:, 0] * x[:, 1]) + (x[:, 2] > 0.3) + 0.1 * rng.normal(size=400)
    quantizer = FeatureQuantizer.fit(x, n_bins=16)
    ensemble = distill_ensemble(x, y, depth, n_trees, quantizer.edges)
    tables = [build_shap_table(tree, x[:30]) for tree in ensemble.trees]
    return quantize_model(ensemble, tables, quantizer, output_bits=8)


def edge_cases(model: QuantizedModel, rng: np.random.Generator, n_random: int) -> np.ndarray:
    """Min, max, codes exactly on every threshold and one above it, then random rows."""
    upper = np.array([e.size for e in model.quantizer.edges])
    rows = [np.zeros_like(upper), upper]
    for tree in model.trees:
        for f, t in zip(tree.features, tree.thresholds, strict=True):
            for code in (t, t + 1):
                row = rng.integers(0, upper + 1)
                row[f] = code
                rows.append(row)
    rows += [rng.integers(0, upper + 1) for _ in range(n_random)]
    return np.array(rows, dtype=np.int64)


@pytest.mark.parametrize(("depth", "n_trees"), [(3, 1), (4, 2), (6, 3)])
def test_simulated_circuit_matches_reference(depth: int, n_trees: int) -> None:
    model = small_model(seed=depth, depth=depth, n_trees=n_trees)
    compiled = compile_model(model)
    codes = edge_cases(model, np.random.default_rng(1), n_random=200)
    for row, want in zip(codes, model.simulate(codes), strict=True):
        got = compiled.expand(np.asarray(compiled.circuit.simulate(row)))[0]
        np.testing.assert_array_equal(got, want)


def test_encrypted_circuit_matches_reference() -> None:
    model = small_model(seed=7, depth=5, n_trees=3)
    compiled = compile_model(model)
    compiled.circuit.keygen()
    codes = edge_cases(model, np.random.default_rng(2), n_random=4)[:12]
    for row, want in zip(codes, model.simulate(codes), strict=True):
        got = compiled.expand(np.asarray(compiled.circuit.encrypt_run_decrypt(row)))[0]
        np.testing.assert_array_equal(got, want)
