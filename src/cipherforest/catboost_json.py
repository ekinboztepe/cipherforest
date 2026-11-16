"""Imports a CatBoost model saved with model.save_model(path, format="json").

CatBoost already trains oblivious trees, so no distillation is needed: each tree
maps to an ObliviousTree (its first split is the lowest leaf-index bit), and the
ensemble reproduces CatBoost's raw score, the log-odds for a classifier.

Only what the encrypted pipeline can serve exactly is accepted: symmetric trees,
one output, numeric and categorical features. Anything else is rejected. Missing
values follow the model's nan_mode: "Min" sends NaN below every border, "Max"
above, and the quantiser encodes them the same way.

A categorical split, one-hot or CTR, is a function of one feature's category, so
it becomes a set: the bit of every known category, plus one for any unseen
category. The bits are read from CatBoost itself (probe_categories), which saves
re-implementing its hashing and counters, and the client only learns the list of
categories, not their statistics. CTRs over combinations of features are rejected
(train with max_ctr_complexity=1).

CatBoost compares inputs as float32 against float32 borders. Inputs that are exact
float32 values route exactly like CatBoost; a float64 input lying between a border
and its float32 neighbour may not.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from cipherforest.oblivious import FloatArray, ObliviousEnsemble, ObliviousTree
from cipherforest.quantize import FeatureQuantizer

UNSEEN = "\x1fcipherforest unseen category"
MAX_LEVELS = 16
# bits[f][code, tree, level]: the split bit for each category code of feature f, -1 where
# it also depends on the rest of the row
CategoryBits = dict[int, np.ndarray]


def probe_categories(model: Any, x_train: object, n_rows: int = 8) -> tuple[dict[int, tuple[str, ...]], CategoryBits]:
    """Each categorical feature's known categories, and every split bit for every category.

    Every category, and one unseen value, is tried on n_rows training rows through
    CatBoost's calc_leaf_indexes. A split that reads only this feature gives the same
    bits on every row.
    """
    from catboost import Pool

    rows = np.array(x_train, dtype=object, ndmin=2)
    cat = [int(i) for i in model.get_cat_feature_indices()]
    for g in cat:
        rows[:, g] = [str(v) for v in rows[:, g]]
    known = {f: tuple(sorted(set(rows[:, f]))) for f in cat}
    base = rows[np.linspace(0, len(rows) - 1, min(n_rows, len(rows))).astype(int)]
    bits: CategoryBits = {}
    for f in cat:
        values = np.array([*known[f], UNSEEN], dtype=object)
        probe = np.repeat(base, len(values), axis=0)
        probe[:, f] = np.tile(values, len(base))
        leaves = np.asarray(model.calc_leaf_indexes(Pool(probe.tolist(), cat_features=cat)), dtype=np.int64)
        level = (leaves[:, :, None] >> np.arange(MAX_LEVELS)) & 1  # (rows * codes, trees, levels)
        level = level.reshape(len(base), len(values), *level.shape[1:])
        same_on_every_row = np.all(level == level[:1], axis=(0, 1))
        moves_with_f = np.any(level != level[:, :1], axis=(0, 1))
        bits[f] = np.where((moves_with_f & ~same_on_every_row)[None], -1, level[0])
    return known, bits


def load_catboost_json(
    text: str,
    extra_edges: FeatureQuantizer | None = None,
    categories: tuple[dict[int, tuple[str, ...]], CategoryBits] | None = None,
) -> tuple[ObliviousEnsemble, FeatureQuantizer]:
    """The quantiser edges are the model's borders, merged with extra_edges if given.

    The JSON keeps only the borders the trees use, so without extra_edges the
    client bundle lists every threshold of the model. Pass training quantiles
    (FeatureQuantizer.fit) to hide them among other values. A model with
    categorical features also needs categories, from probe_categories.
    """
    doc = json.loads(text)
    if "oblivious_trees" not in doc:
        raise ValueError("only symmetric (oblivious) trees are supported; train with grow_policy='SymmetricTree'")
    info = doc["features_info"]
    if info.get("text_features") or info.get("embedding_features"):
        raise ValueError("text and embedding features are not supported")
    cats = info.get("categorical_features") or []
    if cats and categories is None:
        raise ValueError("a model with categorical features needs the CatBoost model and x_train (from_catboost)")
    for ctr in info.get("ctrs", []):
        elements = ctr.get("elements", [])
        if len(elements) != 1 or elements[0].get("combination_element") != "cat_feature_value":
            raise ValueError("CTRs over feature combinations are not supported; train with max_ctr_complexity=1")

    floats = info.get("float_features", [])
    n_features = max([f["flat_feature_index"] for f in floats] + [c["flat_feature_index"] for c in cats]) + 1
    borders: list[FloatArray] = [np.empty(0) for _ in range(n_features)]
    nan_high = [False] * n_features
    flat_index: dict[int, int] = {}
    for f in floats:
        flat_index[f["feature_index"]] = f["flat_feature_index"]
        borders[f["flat_feature_index"]] = np.asarray(f.get("borders", []), dtype=np.float64)
        treatment = f.get("nan_value_treatment", "AsIs")
        if treatment == "AsTrue":
            nan_high[f["flat_feature_index"]] = True
        elif f["has_nans"] and treatment != "AsFalse":
            raise ValueError(f"feature {f['flat_feature_index']}: nan_mode must be Min or Max, not {treatment}")
    cat_flat = {c["feature_index"]: c["flat_feature_index"] for c in cats}

    scale, biases = doc["scale_and_bias"]
    if len(biases) != 1:
        raise ValueError(f"only single-output models are supported, this one has {len(biases)} outputs")

    known, cat_bits = categories or ({}, {})
    trees = []
    for i, t in enumerate(doc["oblivious_trees"]):
        splits = t["splits"]
        leaf_values = np.asarray(t["leaf_values"], dtype=np.float64)
        if leaf_values.size != 1 << len(splits):
            raise ValueError("only single-output models are supported")
        features, thresholds, sets = [], [], []
        for m, s in enumerate(splits):
            kind = s["split_type"]
            if kind == "FloatFeature":
                features.append(flat_index[s["float_feature_index"]])
                thresholds.append(s["border"])
                sets.append(None)
                continue
            if kind == "OneHotFeature":
                f = cat_flat[s["cat_feature_index"]]
            elif kind == "OnlineCtr":
                f = _ctr_feature(cat_bits, i, m)
            else:
                raise ValueError(f"unsupported split type {kind}")
            column = cat_bits[f][:, i, m]
            if np.any(column < 0):
                raise ValueError(f"tree {i} level {m} depends on more than feature {f}")
            features.append(f)
            thresholds.append(0.0)
            sets.append(column.astype(bool))
        trees.append(
            ObliviousTree(
                features=np.array(features, dtype=np.int64),
                thresholds=np.array(thresholds, dtype=np.float64),
                leaf_values=scale * leaf_values,
                sets=tuple(sets) if any(s is not None for s in sets) else (),
            )
        )

    if extra_edges is not None:
        if extra_edges.n_features != n_features:
            raise ValueError(f"extra_edges has {extra_edges.n_features} features, the model {n_features}")
        borders = [np.union1d(b, e) for b, e in zip(borders, extra_edges.edges)]
    categorical: list[tuple[str, ...] | None] = [None] * n_features
    for f, values in known.items():
        categorical[f] = values
        borders[f] = np.arange(len(values), dtype=np.float64) + 0.5  # codes 0..len(values) pass through
    quantizer = FeatureQuantizer(
        tuple(borders), tuple(nan_high) if any(nan_high) else (), tuple(categorical) if known else ()
    )
    return ObliviousEnsemble(trees=trees, bias=float(biases[0])), quantizer


def _ctr_feature(cat_bits: CategoryBits, tree: int, level: int) -> int:
    """The categorical feature a CTR split reads: the one whose categories move its bit."""
    moving = [f for f, b in cat_bits.items() if np.any(b[:, tree, level] != b[0, tree, level])]
    if len(moving) > 1:
        raise ValueError(f"tree {tree} level {level} depends on several categorical features")
    if moving:
        return moving[0]
    # the same bit for every category: any categorical feature gives the same model and SHAP values
    return min(cat_bits)
