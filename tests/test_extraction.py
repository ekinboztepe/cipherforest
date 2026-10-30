from itertools import combinations

import numpy as np
import pytest

from cipherforest.distill import distill_ensemble
from cipherforest.extraction import (
    CellGrid,
    CountingOracle,
    code_values,
    extract,
    find_thresholds,
    maximal_cliques,
    plaintext_oracle,
    prediction_interactions,
    prediction_only,
    reconstruct,
    shap_interactions,
    true_thresholds,
)
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
    for oracle in (plaintext_oracle(model), prediction_only(plaintext_oracle(model))):
        assert find_thresholds(oracle, model.quantizer, 1, np.random.default_rng(0)) <= truth


@pytest.mark.parametrize(("depth", "n_trees"), [(2, 1), (4, 2), (6, 3)])
def test_shap_finds_every_threshold(depth: int, n_trees: int) -> None:
    model = small_model(depth, depth, n_trees)
    found = find_thresholds(plaintext_oracle(model), model.quantizer, 3, np.random.default_rng(0))
    assert found == true_thresholds(model)


def test_query_count() -> None:
    model = small_model(1, 3, 2)
    oracle = CountingOracle(plaintext_oracle(model))
    find_thresholds(oracle, model.quantizer, 1, np.random.default_rng(0))
    # the starting point itself is in every sweep but asked only once
    swept = [e for e in model.quantizer.edges if e.size]
    assert oracle.queries == sum(e.size + 1 for e in swept) - (len(swept) - 1)


def test_repeated_inputs_are_counted_once() -> None:
    model = small_model(1, 3, 2)
    oracle = CountingOracle(plaintext_oracle(model))
    x = random_rows(model, 50, np.random.default_rng(0))
    first = oracle(np.vstack([x, x[:10]]))
    second = oracle(x[::-1])
    assert oracle.queries == 50
    for a, b, c in zip(first, second, plaintext_oracle(model)(x), strict=True):
        np.testing.assert_array_equal(a[:50], c)
        np.testing.assert_array_equal(b[::-1], c)


def tree_graph(model: QuantizedModel) -> set[tuple[int, int]]:
    return {(min(f, g), max(f, g)) for t in model.trees for f, g in combinations({int(v) for v in t.features}, 2)}


def random_rows(model: QuantizedModel, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack([rng.choice(code_values(e), size=n) for e in model.quantizer.edges])


def test_cell_grid_inputs_land_in_their_cells() -> None:
    model = small_model(3, 4, 2)
    grid = CellGrid(model.quantizer, true_thresholds(model))
    cells = np.column_stack([np.random.default_rng(0).integers(0, s, size=200) for s in grid.sizes])
    np.testing.assert_array_equal(grid.index(grid.inputs(cells)), cells)


def test_maximal_cliques() -> None:
    edges = {(0, 1), (1, 2), (0, 2), (2, 3), (4, 5)}
    got = {frozenset(c) for c in maximal_cliques(range(7), edges)}
    assert got == {frozenset({0, 1, 2}), frozenset({2, 3}), frozenset({4, 5}), frozenset({6})}


def test_inclusion_exclusion_rebuilds_a_sum_over_cliques() -> None:
    """Overlapping cliques, an anchor anywhere, and an oracle that is a plain table sum."""
    rng = np.random.default_rng(5)
    quantizer = FeatureQuantizer(tuple(np.arange(3.0) for _ in range(5)))
    grid = CellGrid(quantizer, {(f, k) for f in range(5) for k in range(3)})
    cliques = [frozenset({0, 1, 2}), frozenset({2, 3}), frozenset({1, 2, 4})]
    tables = [rng.normal(size=(4,) * len(c) + (6,)) for c in cliques]

    def oracle(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cells = grid.index(x)
        out = sum(t[tuple(cells[:, f] for f in sorted(c))] for c, t in zip(cliques, tables, strict=True))
        return out[:, 0], out[:, 1:]

    copy = reconstruct(oracle, grid, cliques, anchor=np.array([3, 0, 2, 1, 0]))
    x = grid.inputs(np.column_stack([rng.integers(0, 4, size=300) for _ in range(5)]))
    for got, want in zip(copy(x), oracle(x), strict=True):
        np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.mark.parametrize(("depth", "n_trees"), [(3, 2), (4, 3), (6, 3)])
def test_interaction_graphs_are_subsets_of_the_tree_graph(depth: int, n_trees: int) -> None:
    model = small_model(depth + 10, depth, n_trees)
    grid = CellGrid(model.quantizer, true_thresholds(model))
    oracle = plaintext_oracle(model)
    truth = tree_graph(model)
    assert shap_interactions(oracle, grid, 1, np.random.default_rng(0)) <= truth
    assert prediction_interactions(prediction_only(oracle), grid, 1, np.random.default_rng(0)) <= truth
    assert shap_interactions(oracle, grid, 3, np.random.default_rng(0)) == truth


@pytest.mark.parametrize("use_shap", [True, False])
@pytest.mark.parametrize(("depth", "n_trees"), [(3, 2), (4, 3), (6, 3)])
def test_end_to_end_extraction_is_exact(depth: int, n_trees: int, use_shap: bool) -> None:
    model = small_model(depth + 20, depth, n_trees)
    view = plaintext_oracle(model) if use_shap else prediction_only(plaintext_oracle(model))
    oracle = CountingOracle(view)
    rng = np.random.default_rng(0)
    graph = shap_interactions if use_shap else prediction_interactions
    copy, _ = extract(oracle, model.quantizer, rng, graph)

    x = random_rows(model, 2000, np.random.default_rng(1))
    for got, want in zip(copy(x), view(x), strict=True):
        np.testing.assert_allclose(got, want, atol=1e-9)
    assert oracle.queries < np.prod([e.size + 1 for e in model.quantizer.edges])


def test_full_grid_extraction_is_exact() -> None:
    model = small_model(30, 3, 2)
    view = prediction_only(plaintext_oracle(model))
    copy, _ = extract(CountingOracle(view), model.quantizer, np.random.default_rng(0), graph=None)
    x = random_rows(model, 2000, np.random.default_rng(1))
    np.testing.assert_allclose(copy(x)[0], view(x)[0], atol=1e-9)
