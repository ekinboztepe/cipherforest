"""The leaf-indexed SHAP table against brute-force Shapley values.

The oracle shares no code with shap_table: it works on raw feature vectors and
averages marginal contributions over every ordering of the players.
"""

from collections.abc import Callable
from itertools import permutations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from cipherforest.oblivious import FloatArray, ObliviousEnsemble, ObliviousTree, bits_to_index
from cipherforest.shap_table import _all_bit_vectors, build_shap_table, explain, explain_ensemble

ATOL = 1e-9


def brute_force_shap(
    predict: Callable[[FloatArray], FloatArray],
    x_row: FloatArray,
    background: FloatArray,
    players: list[int],
    n_features: int,
) -> FloatArray:
    def value(coalition: list[int]) -> float:
        hybrid = background.copy()
        hybrid[:, coalition] = x_row[coalition]
        return float(predict(hybrid).mean())

    phi = np.zeros(n_features)
    orders = list(permutations(players))
    for order in orders:
        coalition: list[int] = []
        previous = value(coalition)
        for player in order:
            coalition.append(player)
            current = value(coalition)
            phi[player] += current - previous
            previous = current
    return phi / len(orders)


def random_tree(rng: np.random.Generator, depth: int, n_features: int) -> ObliviousTree:
    return ObliviousTree(
        features=rng.integers(0, n_features, size=depth).astype(np.int64),
        thresholds=rng.uniform(-1.0, 1.0, size=depth),
        leaf_values=rng.uniform(-5.0, 5.0, size=1 << depth),
    )


scenario = st.tuples(
    st.integers(min_value=0, max_value=2**32 - 1),  # seed
    st.integers(min_value=1, max_value=5),  # depth
    st.integers(min_value=1, max_value=6),  # features
    st.integers(min_value=1, max_value=8),  # background rows
)


@settings(max_examples=150, deadline=None)
@given(scenario)
def test_table_matches_brute_force(params: tuple[int, int, int, int]) -> None:
    seed, depth, n_features, n_background = params
    rng = np.random.default_rng(seed)
    tree = random_tree(rng, depth, n_features)
    background = rng.uniform(-1.5, 1.5, size=(n_background, n_features))
    x = rng.uniform(-1.5, 1.5, size=(4, n_features))

    got = explain(tree, build_shap_table(tree, background), x, n_features)

    players = [int(p) for p in np.unique(tree.features)]
    for i in range(x.shape[0]):
        np.testing.assert_allclose(got[i], brute_force_shap(tree.predict, x[i], background, players, n_features), atol=ATOL)


@settings(max_examples=150, deadline=None)
@given(scenario)
def test_efficiency_axiom(params: tuple[int, int, int, int]) -> None:
    seed, depth, n_features, n_background = params
    rng = np.random.default_rng(seed)
    tree = random_tree(rng, depth, n_features)
    background = rng.uniform(-1.5, 1.5, size=(n_background, n_features))
    x = rng.uniform(-1.5, 1.5, size=(16, n_features))

    table = build_shap_table(tree, background)
    phi = explain(tree, table, x, n_features)

    np.testing.assert_allclose(phi.sum(axis=1) + table.expected_value, tree.predict(x), atol=ATOL)
    np.testing.assert_allclose(table.expected_value, tree.predict(background).mean(), atol=ATOL)


@settings(max_examples=60, deadline=None)
@given(scenario, st.integers(min_value=2, max_value=3))
def test_ensemble_is_sum_of_tree_tables(params: tuple[int, int, int, int], n_trees: int) -> None:
    seed, depth, n_features, n_background = params
    rng = np.random.default_rng(seed)
    ensemble = ObliviousEnsemble(trees=[random_tree(rng, depth, n_features) for _ in range(n_trees)], bias=0.7)
    background = rng.uniform(-1.5, 1.5, size=(n_background, n_features))
    x = rng.uniform(-1.5, 1.5, size=(3, n_features))

    tables = [build_shap_table(tree, background) for tree in ensemble.trees]
    got = explain_ensemble(ensemble, tables, x, n_features)

    players = sorted({int(f) for tree in ensemble.trees for f in tree.features})
    for i in range(x.shape[0]):
        want = brute_force_shap(ensemble.predict, x[i], background, players, n_features)
        np.testing.assert_allclose(got[i], want, atol=ATOL)


def test_bit_vectors_enumerate_leaves_in_index_order() -> None:
    for depth in range(1, 7):
        np.testing.assert_array_equal(bits_to_index(_all_bit_vectors(depth)), np.arange(1 << depth))
