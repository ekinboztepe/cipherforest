"""Imports a CatBoost model saved with model.save_model(path, format="json").

CatBoost already trains oblivious trees, so no distillation is needed: each tree
maps to an ObliviousTree (its first split is the lowest leaf-index bit), and the
ensemble reproduces CatBoost's raw score, the log-odds for a classifier.

Only what the encrypted pipeline can serve exactly is accepted: symmetric trees,
one output, numeric features without NaNs. Anything else is rejected.

CatBoost compares inputs as float32 against float32 borders. Inputs that are exact
float32 values route exactly like CatBoost; a float64 input lying between a border
and its float32 neighbour may not.
"""

from __future__ import annotations

import json

import numpy as np

from cipherforest.oblivious import FloatArray, ObliviousEnsemble, ObliviousTree
from cipherforest.quantize import FeatureQuantizer


def load_catboost_json(
    text: str, extra_edges: FeatureQuantizer | None = None
) -> tuple[ObliviousEnsemble, FeatureQuantizer]:
    """The quantiser edges are the model's borders, merged with extra_edges if given.

    The JSON keeps only the borders the trees use, so without extra_edges the
    client bundle lists every threshold of the model. Pass training quantiles
    (FeatureQuantizer.fit) to hide them among other values.
    """
    doc = json.loads(text)
    if "oblivious_trees" not in doc:
        raise ValueError("only symmetric (oblivious) trees are supported; train with grow_policy='SymmetricTree'")
    info = doc["features_info"]
    if info.get("categorical_features") or info.get("text_features") or info.get("embedding_features"):
        raise ValueError("only numeric features are supported")

    floats = info["float_features"]
    if any(f["has_nans"] for f in floats):
        raise ValueError("features with NaNs are not supported")
    n_features = max(f["flat_feature_index"] for f in floats) + 1
    borders: list[FloatArray] = [np.empty(0) for _ in range(n_features)]
    flat_index: dict[int, int] = {}
    for f in floats:
        flat_index[f["feature_index"]] = f["flat_feature_index"]
        borders[f["flat_feature_index"]] = np.asarray(f["borders"], dtype=np.float64)

    scale, biases = doc["scale_and_bias"]
    if len(biases) != 1:
        raise ValueError(f"only single-output models are supported, this one has {len(biases)} outputs")

    trees = []
    for t in doc["oblivious_trees"]:
        splits = t["splits"]
        if any(s["split_type"] != "FloatFeature" for s in splits):
            raise ValueError("only float feature splits are supported")
        leaf_values = np.asarray(t["leaf_values"], dtype=np.float64)
        if leaf_values.size != 1 << len(splits):
            raise ValueError("only single-output models are supported")
        trees.append(
            ObliviousTree(
                features=np.array([flat_index[s["float_feature_index"]] for s in splits], dtype=np.int64),
                thresholds=np.array([s["border"] for s in splits], dtype=np.float64),
                leaf_values=scale * leaf_values,
            )
        )

    if extra_edges is not None:
        if extra_edges.n_features != n_features:
            raise ValueError(f"extra_edges has {extra_edges.n_features} features, the model {n_features}")
        borders = [np.union1d(b, e) for b, e in zip(borders, extra_edges.edges)]
    return ObliviousEnsemble(trees=trees, bias=float(biases[0])), FeatureQuantizer(tuple(borders))
