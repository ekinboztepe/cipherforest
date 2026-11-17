"""0.3 on a rented server: CatBoost served with secret and public splits, against Concrete-ML.

Runs on a Runpod CPU pod with cipherforest 0.3.0 installed from PyPI (no repository
code), next to Concrete-ML 1.9 in the same environment, so both use the same Concrete.
One job per process, each printing a RESULT line:

    python server_v3_bench.py <dataset> <variant> [requests]

Variants: secret-shap, public-shap, public-score (CatBoost depth 4 x 30, seed 0),
xgb-4, xgb-6 (Concrete-ML XGBClassifier, 30 trees, n_bits=6). Every encrypted answer
is checked: ours against cipherforest.explain, Concrete-ML against its simulation.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

OPENML = {"adult": 1590, "bank_marketing": 1461, "credit_default": 42477}
WORK = Path("/tmp/v3")


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    # as experiments/datasets.py: ordinal codes for categories, median for missing numbers
    try:
        bunch = fetch_openml(data_id=OPENML[name], as_frame=True, parser="auto")
    except TypeError:
        bunch = fetch_openml(data_id=OPENML[name], as_frame=True, parser="liac-arff")
    columns = []
    for c in bunch.data.columns:
        col = bunch.data[c]
        if isinstance(col.dtype, pd.CategoricalDtype) or col.dtype == object:
            columns.append(col.astype("category").cat.codes.to_numpy(dtype=np.float64))
        else:
            v = col.to_numpy(dtype=np.float64)
            columns.append(np.where(np.isnan(v), np.nanmedian(v), v))
    target = bunch.target.astype(str)
    return np.column_stack(columns), (target == sorted(target.unique())[-1]).to_numpy(dtype=np.int64)


def ours(x_tr, y_tr, x_te, y_te, rows, public: bool, output: str) -> dict[str, object]:
    import cipherforest
    from catboost import CatBoostClassifier

    cb = CatBoostClassifier(iterations=30, depth=4, verbose=0, random_seed=0, thread_count=4,
                            allow_writing_files=False).fit(x_tr, y_tr)
    background = x_tr[np.random.default_rng(0).choice(len(x_tr), size=100, replace=False)]
    model = cipherforest.from_catboost(cb, background=background, x_train=x_tr)
    score, _ = cipherforest.explain(model, x_te)
    compiled = cipherforest.compile_model(model, output=output, public_splits=public)
    cipherforest.export(compiled, model, WORK)
    server = cipherforest.Server(WORK / "server.zip")
    client = cipherforest.Client.load(WORK / "client.json")
    serialized = client.evaluation_keys()
    keys = server.load_keys(serialized)
    want, want_shap = cipherforest.explain(model, rows)
    run_s = []
    for row, w, phi in zip(rows, want, want_shap, strict=True):
        ciphertext = client.encrypt(row)
        start = time.perf_counter()
        result = server.run(ciphertext, keys)
        run_s.append(time.perf_counter() - start)
        got, got_shap = client.decrypt(result)
        assert abs(got - w) < 1e-9 and (output != "shap" or np.allclose(got_shap, phi)), "differs from explain"
    return {"auc": float(roc_auc_score(y_te, score)), "pbs": compiled.circuit.programmable_bootstrap_count,
            "evaluation_keys_mb": len(serialized) / 2**20, "run_s": run_s}


def xgb(x_tr, y_tr, x_te, y_te, rows, depth: int) -> dict[str, object]:
    from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
    from concrete.ml.sklearn import XGBClassifier

    model = XGBClassifier(n_bits=6, n_estimators=30, max_depth=depth, random_state=0, n_jobs=1).fit(x_tr, y_tr)
    model.compile(x_tr)
    auc = float(roc_auc_score(y_te, model.predict_proba(x_te, fhe="simulate")[:, 1]))
    FHEModelDev(str(WORK / "model"), model).save()
    server = FHEModelServer(str(WORK / "model"))
    server.load()
    client = FHEModelClient(str(WORK / "model"), key_dir=str(WORK / "keys"))
    client.generate_private_and_evaluation_keys()
    keys = client.get_serialized_evaluation_keys()
    want = model.predict_proba(rows, fhe="simulate")
    run_s = []
    for row, w in zip(rows, want, strict=True):
        ciphertext = client.quantize_encrypt_serialize(row[None])
        start = time.perf_counter()
        result = server.run(ciphertext, keys)
        run_s.append(time.perf_counter() - start)
        assert np.abs(client.deserialize_decrypt_dequantize(result)[0] - w).max() < 1e-6, "differs from simulation"
    return {"auc": auc, "pbs": model.fhe_circuit.programmable_bootstrap_count,
            "evaluation_keys_mb": len(keys) / 2**20, "run_s": run_s}


if __name__ == "__main__":
    dataset, variant = sys.argv[1], sys.argv[2]
    requests = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    x, y = load(dataset)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=0)
    rows = x_te[np.random.default_rng(0).choice(len(x_te), size=requests, replace=False)]
    shutil.rmtree(WORK, ignore_errors=True)
    start = time.perf_counter()
    if variant.startswith("xgb"):
        result = xgb(x_tr, y_tr, x_te, y_te, rows, int(variant.split("-")[1]))
    else:
        splits, output = variant.split("-")
        result = ours(x_tr, y_tr, x_te, y_te, rows, splits == "public", output)
    run_s = result.pop("run_s")
    print("RESULT", json.dumps({"dataset": dataset, "variant": variant, "requests": requests, **result,
                                "server_run_s_median": float(np.median(run_s)), "first_s": run_s[0],
                                "run_s": run_s, "job_s": time.perf_counter() - start}), flush=True)
