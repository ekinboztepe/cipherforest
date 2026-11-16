"""The short path from a trained model to the integer model the circuit serves.

    model = from_catboost(cb_model, background=x_train[:100], x_train=x_train)
    pred, shap = explain(model, x_test)            # in the clear, what the client will decrypt
    compiled = compile_model(model)                # needs concrete-python (Linux/macOS)
    export(compiled, model, Path("artifacts"))     # server.zip for the server, client.json for clients

Everything here runs without Concrete, so models can be prepared and checked on any OS.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from cipherforest.align import align_tables
from cipherforest.catboost_json import load_catboost_json, probe_categories
from cipherforest.distill import distill_ensemble
from cipherforest.oblivious import FloatArray, ObliviousEnsemble
from cipherforest.quantize import FeatureQuantizer, QuantizedModel, quantize_model
from cipherforest.shap_table import build_shap_table


def from_ensemble(
    ensemble: ObliviousEnsemble, quantizer: FeatureQuantizer, background: FloatArray, output_bits: int = 8
) -> QuantizedModel:
    """Exact interventional SHAP tables against `background`, then integer outputs.

    Every split must sit on one of the quantiser's edges. The background set is the
    reference SHAP compares against; 100 training rows are usual.
    """
    background = quantizer.fill(background)
    tables = [build_shap_table(tree, background) for tree in ensemble.trees]
    return quantize_model(ensemble, tables, quantizer, output_bits)


def from_catboost(
    model: Any, background: object, x_train: object = None, n_bins: int = 32, output_bits: int = 8
) -> QuantizedModel:
    """A CatBoost model (or the text of model.save_model(..., format="json")), served as is.

    Symmetric trees, one output, numeric and categorical features; NaN follows the
    model's nan_mode (Min or Max). The prediction is CatBoost's raw score, the
    log-odds for a classifier. With x_train, n_bins training quantiles are added to
    the edges, so client.json does not list only the model's own thresholds.

    Categorical features need the model object and x_train: the categories seen in
    x_train are the ones the client can name, any other value is "unseen", and the
    split bits are read from CatBoost (catboost_json.probe_categories). Rows may then
    be lists, object arrays or DataFrames, with categories as strings or integers.
    """
    categories = None
    cat: list[int] = []
    if isinstance(model, str):
        text = model
    else:
        cat = [int(i) for i in model.get_cat_feature_indices()]
        if cat and x_train is None:
            raise ValueError("a model with categorical features needs x_train")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            if cat:
                from catboost import Pool

                rows = np.array(x_train, dtype=object, ndmin=2)
                for f in cat:
                    rows[:, f] = [str(v) for v in rows[:, f]]
                model.save_model(str(path), format="json", pool=Pool(rows.tolist(), cat_features=cat))
                categories = probe_categories(model, rows)
            else:
                model.save_model(str(path), format="json")
            text = path.read_text()
    padding = None
    if x_train is not None:
        numeric = np.array(x_train, dtype=object, ndmin=2)
        numeric[:, cat] = 0.0
        padding = FeatureQuantizer.fit(numeric.astype(np.float64), n_bins)
    ensemble, quantizer = load_catboost_json(text, extra_edges=padding, categories=categories)
    return from_ensemble(ensemble, quantizer, background, output_bits)


def distill(
    x_train: FloatArray,
    target: FloatArray,
    depth: int = 6,
    n_trees: int = 3,
    n_bins: int = 32,
    background: int | FloatArray = 100,
    output_bits: int = 8,
    seed: int = 0,
    teacher_shap: tuple[FloatArray, FloatArray] | None = None,
    ridge: float = 1.0,
) -> QuantizedModel:
    """Fits a small oblivious ensemble to `target`: labels, or any model's scores on x_train.

    Use it for models the circuit cannot take directly (XGBoost, LightGBM, forests):
    pass their predict_proba on x_train. The copy explains itself exactly, not the
    original, and agrees with it only as well as its size allows.

    teacher_shap=(rows, values) instead divides each leaf's total among the tree's
    features the way the original's SHAP values on those rows do (align.align_tables).
    The prediction is unchanged and the values still sum to it, but they are then no
    longer the copy's own Shapley values. Compute the original's SHAP values against
    the same background.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    quantizer = FeatureQuantizer.fit(x_train, n_bins)
    x_train = quantizer.fill(x_train)  # missing values below every edge, as CatBoost's default
    ensemble = distill_ensemble(x_train, np.asarray(target, dtype=np.float64), depth, n_trees, quantizer.edges)
    if isinstance(background, int):
        rng = np.random.default_rng(seed)
        background = x_train[rng.choice(len(x_train), size=min(background, len(x_train)), replace=False)]
    if teacher_shap is None:
        return from_ensemble(ensemble, quantizer, background, output_bits)
    background = quantizer.fill(np.asarray(background, dtype=np.float64))
    tables = [build_shap_table(tree, background) for tree in ensemble.trees]
    tables = align_tables(ensemble, tables, quantizer.fill(teacher_shap[0]), teacher_shap[1], ridge=ridge)
    return quantize_model(ensemble, tables, quantizer, output_bits)


def explain(model: QuantizedModel, x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """(predictions, SHAP values) in the clear, bit for bit what an encrypted run returns."""
    return model.decode(model.simulate(model.quantizer.transform(np.array(x, dtype=object, ndmin=2))))
