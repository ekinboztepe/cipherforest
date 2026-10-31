import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cipherforest.distill import distill_ensemble
from cipherforest.oblivious import ObliviousEnsemble, bits_to_index
from cipherforest.quantize import FeatureQuantizer, QuantizedModel, _sum_preserving_round, quantize_model, to_grid
from cipherforest.shap_table import ShapTable, build_shap_table, explain_ensemble

scenario = st.tuples(
    st.integers(min_value=0, max_value=2**32 - 1),  # seed
    st.integers(min_value=1, max_value=5),  # depth
    st.integers(min_value=1, max_value=3),  # trees
    st.integers(min_value=1, max_value=6),  # features
    st.integers(min_value=6, max_value=12),  # output bits
)

Params = tuple[int, int, int, int, int]


def build(params: Params) -> tuple[np.ndarray, ObliviousEnsemble, list[ShapTable], QuantizedModel]:
    """Distil a random target and quantise it. Eval inputs include every edge value itself."""
    seed, depth, n_trees, n_features, output_bits = params
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(300, n_features))
    x[:, 0] = np.round(x[:, 0])  # lots of ties
    y = np.sin(2 * x[:, 0]) + (x[:, -1] > 0.2) + 0.1 * rng.normal(size=300)
    quantizer = FeatureQuantizer.fit(x, n_bins=16)
    ensemble = distill_ensemble(x, y, depth, n_trees, quantizer.edges)
    background = x[rng.choice(300, size=20, replace=False)]
    tables = [build_shap_table(tree, background) for tree in ensemble.trees]
    model = quantize_model(ensemble, tables, quantizer, output_bits)

    on_edges = np.column_stack([rng.choice(e, size=50) if e.size else np.zeros(50) for e in quantizer.edges])
    x_eval = np.vstack([rng.normal(size=(200, n_features)), on_edges])
    return x_eval, ensemble, tables, model


@settings(max_examples=100, deadline=None)
@given(scenario)
def test_integer_routing_matches_float_routing(params: Params) -> None:
    x_eval, ensemble, _, model = build(params)
    codes = model.quantizer.transform(x_eval)
    for float_tree, int_tree in zip(ensemble.trees, model.trees, strict=True):
        np.testing.assert_array_equal(int_tree.leaf_index(codes), float_tree.leaf_index(x_eval))


@settings(max_examples=100, deadline=None)
@given(scenario)
def test_rounding_preserves_each_leaf_total(params: Params) -> None:
    _, ensemble, tables, model = build(params)
    for tree, table, qtree in zip(ensemble.trees, tables, model.trees, strict=True):
        totals = np.rint((tree.leaf_values - table.expected_value) * model.scale).astype(np.int64)
        np.testing.assert_array_equal(qtree.shap.sum(axis=1), totals)
        assert np.all(np.abs(qtree.shap - table.values * model.scale) < 1.0)


@settings(max_examples=100, deadline=None)
@given(scenario)
def test_decoded_outputs_stay_within_the_rounding_bound(params: Params) -> None:
    # prediction error <= T / 2s, per-feature SHAP error < T / s
    x_eval, ensemble, tables, model = build(params)
    prediction, shap = model.decode(model.simulate(model.quantizer.transform(x_eval)))

    n_trees = len(model.trees)
    np.testing.assert_array_less(np.abs(prediction - ensemble.predict(x_eval)), n_trees / (2 * model.scale) + 1e-9)
    exact = explain_ensemble(ensemble, tables, x_eval, model.quantizer.n_features)
    assert np.all(np.abs(shap - exact) < n_trees / model.scale)


@settings(max_examples=100, deadline=None)
@given(scenario)
def test_worst_case_output_fits_the_declared_width(params: Params) -> None:
    _, _, _, model = build(params)
    worst = np.zeros(model.quantizer.n_features, dtype=np.int64)
    for tree in model.trees:
        worst[tree.players] += np.abs(tree.shap).max(axis=0)
    assert worst.max() <= (1 << (model.output_bits - 1)) - 1


@settings(max_examples=30, deadline=None)
@given(scenario)
def test_json_round_trip(params: Params) -> None:
    x_eval, _, _, model = build(params)
    restored = QuantizedModel.from_json(model.to_json())
    codes = model.quantizer.transform(x_eval)
    np.testing.assert_array_equal(restored.quantizer.transform(x_eval), codes)
    np.testing.assert_array_equal(restored.simulate(codes), model.simulate(codes))
    assert (restored.scale, restored.base_value, restored.output_bits) == (
        model.scale,
        model.base_value,
        model.output_bits,
    )


def test_sum_preserving_round_with_negative_remainder() -> None:
    values = np.array([[0.9, 0.9, -0.2], [1.5, -1.5, 0.0]])
    rounded = _sum_preserving_round(values, np.array([1, 0]))
    np.testing.assert_array_equal(rounded.sum(axis=1), [1, 0])
    assert np.all(np.abs(rounded - values) < 1.0)


def test_threshold_must_be_an_edge() -> None:
    quantizer = FeatureQuantizer(edges=(np.array([0.0, 1.0, 2.0]),))
    assert quantizer.threshold_index(0, 1.0) == 1
    with pytest.raises(ValueError, match="not an edge"):
        quantizer.threshold_index(0, 1.5)


def test_codes_count_edges_strictly_below() -> None:
    quantizer = FeatureQuantizer(edges=(np.array([0.0, 1.0, 2.0]),))
    x = np.array([[-1.0], [0.0], [0.5], [1.0], [2.0], [3.0]])
    np.testing.assert_array_equal(quantizer.transform(x)[:, 0], [0, 0, 1, 1, 2, 3])
    assert quantizer.input_bits == 2


@settings(max_examples=30, deadline=None)
@given(scenario, st.integers(min_value=3, max_value=10))
def test_grid_model_routes_on_the_next_grid_edge_up(params: Params, bits: int) -> None:
    x_eval, _, _, model = build(params)
    # the owner's range covers the training data, so every edge
    low = np.array([min(x_eval[:, f].min(), *e) for f, e in enumerate(model.quantizer.edges)])
    high = np.array([max(x_eval[:, f].max(), *e) for f, e in enumerate(model.quantizer.edges)])
    grid = to_grid(model, low, high, bits)
    assert grid.quantizer.input_bits <= bits
    codes = grid.quantizer.transform(x_eval)
    step = (high - low) / (1 << bits)
    for tree, moved in zip(model.trees, grid.trees, strict=True):
        before = np.array([model.quantizer.edges[f][t] for f, t in zip(tree.features, tree.thresholds, strict=True)])
        # past the last grid edge the comparison is never true
        after = np.array([np.append(grid.quantizer.edges[f], np.inf)[t] for f, t in zip(moved.features, moved.thresholds, strict=True)])
        top = np.array([np.append(grid.quantizer.edges[f], -np.inf)[-1] for f in tree.features])
        assert np.all((after >= before) & ((after - before <= step[tree.features] + 1e-12) | (before > top)))
        np.testing.assert_array_equal(moved.leaf_index(codes), bits_to_index(x_eval[:, tree.features] > after))
