from itertools import combinations

import numpy as np
import pytest

from cipherforest.distill import distill_ensemble
from cipherforest.extraction import (
    bisect_edges,
    Cached,
    CellGrid,
    Counted,
    code_values,
    denoised,
    extract,
    extract_hidden,
    fit_tables,
    find_thresholds,
    maximal_cliques,
    noisy_oracle,
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
    counter = Counted(plaintext_oracle(model))
    find_thresholds(Cached(counter), model.quantizer, 1, np.random.default_rng(0))
    # the starting point itself is in every sweep but asked only once
    swept = [e for e in model.quantizer.edges if e.size]
    assert counter.queries == sum(e.size + 1 for e in swept) - (len(swept) - 1)


def test_repeated_inputs_are_counted_once() -> None:
    model = small_model(1, 3, 2)
    counter = Counted(plaintext_oracle(model))
    oracle = Cached(counter)
    x = random_rows(model, 50, np.random.default_rng(0))
    first = oracle(np.vstack([x, x[:10]]))
    second = oracle(x[::-1])
    assert counter.queries == 50
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
    counter = Counted(view)
    rng = np.random.default_rng(0)
    graph = shap_interactions if use_shap else prediction_interactions
    copy, _ = extract(Cached(counter), model.quantizer, rng, graph)
    assert copy is not None

    x = random_rows(model, 2000, np.random.default_rng(1))
    for got, want in zip(copy(x), view(x), strict=True):
        np.testing.assert_allclose(got, want, atol=1e-9)
    assert counter.queries < np.prod([e.size + 1 for e in model.quantizer.edges])


def test_full_grid_extraction_is_exact() -> None:
    model = small_model(30, 3, 2)
    view = prediction_only(plaintext_oracle(model))
    copy, _ = extract(Cached(view), model.quantizer, np.random.default_rng(0), graph=None)
    assert copy is not None
    x = random_rows(model, 2000, np.random.default_rng(1))
    np.testing.assert_allclose(copy(x)[0], view(x)[0], atol=1e-9)


def test_noise_is_on_the_integer_grid_and_only_on_active_features() -> None:
    model = small_model(40, 3, 2, n_features=8)
    x = random_rows(model, 500, np.random.default_rng(0))
    pred, shap = noisy_oracle(model, 2.0, np.random.default_rng(1))(x)
    clean_pred, clean_shap = plaintext_oracle(model)(x)
    eta = (shap - clean_shap) * model.scale
    np.testing.assert_allclose(eta, np.rint(eta), atol=1e-9)
    inactive = np.setdiff1d(np.arange(8), np.concatenate([t.players for t in model.trees]))
    assert inactive.size and np.all(shap[:, inactive] == 0)
    np.testing.assert_allclose(pred - clean_pred, eta.sum(axis=1) / model.scale, atol=1e-9)
    assert np.any(eta != 0)


def test_enough_repeats_remove_the_noise() -> None:
    model = small_model(41, 4, 2)
    x = random_rows(model, 300, np.random.default_rng(0))
    server = noisy_oracle(model, 1.0, np.random.default_rng(1))
    view = denoised(server, 2000, model.scale, model.base_value, batch=50_000)
    for got, want in zip(view(x), plaintext_oracle(model)(x), strict=True):
        np.testing.assert_allclose(got, want, atol=1e-9)
    _, shap = denoised(server, 1, model.scale, model.base_value)(x)
    assert not np.allclose(shap, plaintext_oracle(model)(x)[1])


def test_extraction_through_noise_is_exact_with_enough_repeats() -> None:
    model = small_model(42, 3, 2)
    counter = Counted(noisy_oracle(model, 0.5, np.random.default_rng(1)))
    oracle = Cached(denoised(counter, 400, model.scale, model.base_value))
    copy, _ = extract(oracle, model.quantizer, np.random.default_rng(0), shap_interactions)
    assert copy is not None
    x = random_rows(model, 2000, np.random.default_rng(2))
    for got, want in zip(copy(x), plaintext_oracle(model)(x), strict=True):
        np.testing.assert_allclose(got, want, atol=1e-9)
    assert counter.queries % 400 == 0


def test_too_few_repeats_give_up_instead_of_returning_a_wrong_copy() -> None:
    model = small_model(43, 4, 3)
    oracle = Cached(denoised(noisy_oracle(model, 4.0, np.random.default_rng(1)), 1, model.scale, model.base_value))
    copy, _ = extract(oracle, model.quantizer, np.random.default_rng(0), shap_interactions, max_cells=5000)
    assert copy is None


def test_cell_budget_does_not_overflow() -> None:
    # 40^8 cells overflows a 32-bit product, which np.prod uses on Windows
    quantizer = FeatureQuantizer(tuple(np.arange(39.0) for _ in range(8)))
    grid = CellGrid(quantizer, {(f, k) for f in range(8) for k in range(39)})
    with pytest.raises(ValueError, match="budget"):
        reconstruct(plaintext_oracle, grid, [frozenset(range(8))], np.zeros(8, dtype=np.int64), max_cells=1000)


def fitted_copy(model: QuantizedModel, oracle, n: int, rng: np.random.Generator):
    grid = CellGrid(model.quantizer, true_thresholds(model))
    cliques = maximal_cliques(range(grid.sizes.size), tree_graph(model))
    # uniform over cells, not codes, so that no cell is rare
    x = grid.inputs(np.column_stack([rng.integers(0, s, size=n) for s in grid.sizes]))
    return fit_tables(grid, cliques, x, np.column_stack(oracle(x)))


def test_fitted_tables_are_exact_without_noise() -> None:
    model = small_model(50, 4, 3)
    copy = fitted_copy(model, plaintext_oracle(model), 3000, np.random.default_rng(0))
    x = random_rows(model, 2000, np.random.default_rng(1))
    for got, want in zip(copy(x), plaintext_oracle(model)(x), strict=True):
        np.testing.assert_allclose(got, want, atol=1e-9)


def test_fitted_tables_round_back_to_the_model_through_noise() -> None:
    model = small_model(51, 3, 2)
    copy = fitted_copy(model, noisy_oracle(model, 0.5, np.random.default_rng(2)), 200_000, np.random.default_rng(0))
    x = random_rows(model, 2000, np.random.default_rng(1))
    _, shap = copy(x)
    np.testing.assert_allclose(np.rint(shap * model.scale) / model.scale, plaintext_oracle(model)(x)[1], atol=1e-9)


def used_edges(model: QuantizedModel) -> set[tuple[int, float]]:
    return {(f, float(model.quantizer.edges[f][k])) for f, k in true_thresholds(model)}


def feature_range(model: QuantizedModel) -> tuple[np.ndarray, np.ndarray]:
    edges = model.quantizer.edges
    return np.array([e[0] - 1 for e in edges]), np.array([e[-1] + 1 for e in edges])


@pytest.mark.parametrize(("depth", "n_trees"), [(2, 1), (4, 2), (6, 3)])
def test_bisection_recovers_used_edges_bit_for_bit(depth: int, n_trees: int) -> None:
    model = small_model(depth + 60, depth, n_trees)
    low, high = feature_range(model)
    starts = np.random.default_rng(0).uniform(low, high, size=(3, low.size))
    empty = [np.zeros(0) for _ in low]
    for oracle in (plaintext_oracle(model), prediction_only(plaintext_oracle(model))):
        found = bisect_edges(oracle, low, high, starts, empty)
        assert {(f, float(e)) for f, es in enumerate(found) for e in es} <= used_edges(model)
    found = bisect_edges(plaintext_oracle(model), low, high, starts, empty)
    assert {(f, float(e)) for f, es in enumerate(found) for e in es} == used_edges(model)


def test_known_edges_are_not_bisected_again() -> None:
    model = small_model(64, 4, 2)
    low, high = feature_range(model)
    start = np.random.default_rng(0).uniform(low, high, size=(1, low.size))
    first = bisect_edges(plaintext_oracle(model), low, high, start, [np.zeros(0) for _ in low])
    counter = Counted(plaintext_oracle(model))
    again = bisect_edges(counter, low, high, start, first)
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    sweep = sum(np.unique(np.concatenate([np.linspace(lo, hi, 16), e, np.nextafter(e, np.inf)])).size - 1
                for lo, hi, e in zip(low, high, first, strict=True))
    assert counter.queries == 2 * sweep  # both ends of each interval, no bisection


@pytest.mark.parametrize("use_shap", [True, False])
def test_extraction_with_hidden_edges_is_exact(use_shap: bool) -> None:
    model = small_model(65, 4, 3)
    low, high = feature_range(model)
    view = plaintext_oracle(model) if use_shap else prediction_only(plaintext_oracle(model))
    graph = shap_interactions if use_shap else prediction_interactions
    copy, _ = extract_hidden(Cached(view), low, high, np.random.default_rng(0), graph)
    assert copy is not None
    rng = np.random.default_rng(1)
    for x in (rng.uniform(low, high, size=(2000, low.size)), random_rows(model, 2000, rng)):
        for got, want in zip(copy(x), view(x), strict=True):
            np.testing.assert_allclose(got, want, atol=1e-9)
