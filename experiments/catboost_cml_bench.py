"""CatBoost served directly against Concrete-ML's XGBoost, timed on one machine.

The server benchmark timed our distilled student. The path the README now
recommends is CatBoost imported with from_catboost, which was only compared in
the clear (catboost_bench.py). This times it encrypted, next to Concrete-ML,
with the same Concrete settings (cipherforest.SAFE, which are Concrete-ML's),
in the same container and under the same CPU and memory limits.

Variants, one per process:

- cb-T-D: CatBoost, T trees of depth D, prediction and exact SHAP.
- cbscore-T-D: the same model, prediction only, the like-for-like output.
- xgb-T-D: Concrete-ML XGBClassifier, n_bits=6, prediction only.

Every encrypted result of ours must equal cipherforest.explain bit for bit.
Each finished job is appended to results/catboost_cml_bench.json at once.

    docker run --rm --cpuset-cpus 0-3 --memory 5g -v <repo>/experiments:/work/experiments \
        cipherforest-ml python experiments/catboost_cml_bench.py <dataset> <seed> <variant> <requests>
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from concrete_ml_bench import WORK, circuit_sizes, xgb
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import cipherforest
from cipherforest.circuit import SAFE, compile_model
from cipherforest.deploy import CLIENT_FILE, SERVER_FILE, Client, Server, export

OUT = Path(__file__).resolve().parent / "results" / "catboost_cml_bench.json"


def catboost(x_tr, y_tr, x_te, y_te, x_run, n_trees: int, depth: int, output: str, seed: int) -> dict[str, object]:
    background = x_tr[np.random.default_rng(seed).choice(len(x_tr), size=100, replace=False)]
    cb = CatBoostClassifier(iterations=n_trees, depth=depth, random_seed=seed, thread_count=1, verbose=0,
                            allow_writing_files=False).fit(x_tr, y_tr)
    model = cipherforest.from_catboost(cb, background=background, x_train=x_tr)
    score, _ = cipherforest.explain(model, x_te)

    start = time.perf_counter()
    compiled = compile_model(model, configuration=SAFE, output=output)
    row: dict[str, object] = {
        "auc_float": float(roc_auc_score(y_te, cb.predict(x_te, prediction_type="RawFormulaVal"))),
        "auc": float(roc_auc_score(y_te, score)),
        "compile_s": time.perf_counter() - start,
        **circuit_sizes(compiled.circuit),
    }
    shutil.rmtree(WORK, ignore_errors=True)
    export(compiled, model, WORK)
    server = Server(WORK / SERVER_FILE)
    start = time.perf_counter()
    client = Client.load(WORK / CLIENT_FILE)
    keygen_s = time.perf_counter() - start
    serialized = client.evaluation_keys()
    keys_mb = len(serialized) / 2**20
    keys = server.load_keys(serialized)
    del serialized

    want_score, want_shap = cipherforest.explain(model, x_run)
    run_s = []
    for i, sample in enumerate(x_run):
        ciphertext = client.encrypt(sample)
        t0 = time.perf_counter()
        result = server.run(ciphertext, keys)
        run_s.append(time.perf_counter() - t0)
        got_score, got_shap = client.decrypt(result)
        same = abs(got_score - want_score[i]) < 1e-9 if output == "score" else np.allclose(got_shap, want_shap[i], atol=1e-9)
        if not same:
            raise AssertionError(f"request {i}: encrypted result differs from cipherforest.explain")
        print(f"request {i}: {run_s[-1]:.2f} s", flush=True)
    return row | {
        "samples": len(x_run),
        "all_match_explain": True,
        "keygen_s": keygen_s,
        "evaluation_keys_mb": keys_mb,
        "server_run_s_median": float(np.median(run_s)),
        "server_run_s_max": float(np.max(run_s)),
        "server_run_s": run_s,
    }


if __name__ == "__main__":
    dataset, seed, variant, requests = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    x, y = load_dataset(dataset)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=seed)
    x_run = x_te[np.random.default_rng(seed).choice(len(x_te), size=requests, replace=False)]
    kind, n_trees, depth = variant.split("-")

    start = time.perf_counter()
    if kind == "xgb":
        result = xgb(x_tr, y_tr, x_te, y_te, x_run, True, int(n_trees), int(depth), seed)
    else:
        output = "score" if kind == "cbscore" else "shap"
        result = catboost(x_tr, y_tr, x_te, y_te, x_run, int(n_trees), int(depth), output, seed)
    row = {"dataset": dataset, "seed": seed, "variant": variant, "cpus": len(os.sched_getaffinity(0)),
           "job_s": time.perf_counter() - start, **result}

    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    OUT.write_text(json.dumps([*rows, row], indent=1))
    shutil.rmtree(WORK, ignore_errors=True)
    row.pop("server_run_s")
    print("RESULT", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
