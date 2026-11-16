"""How wrong is the served probability, and what does more output precision cost?

Every SHAP value is rounded to output_bits, and each tree can add up to 1/(2s) to the
prediction, s being the largest scale at which every sum fits. With 30 trees and 8
bits that is up to about 0.3 in log-odds, several points of probability, which AUC
does not show. In the clear (default): CatBoost depth 4 x 30 on the five datasets,
seed 0, largest error over the test split in log-odds and in probability, at 8, 10
and 12 bits. With --time BITS OUTPUT (Docker, docker/Dockerfile plus catboost and
scikit-learn): Adult, 20 encrypted requests through the exported files on SAFE.

    uv run python experiments/precision_bench.py
    python experiments/precision_bench.py --time 10 shap
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import cipherforest

OUT = Path(__file__).resolve().parent / "results" / "precision_bench.json"
TIMES = OUT.with_name("precision_time.json")
DATASETS = ("adult", "credit_default", "bank_marketing", "breast_cancer", "credit_g")
BITS = (8, 10, 12)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def setup(dataset: str):
    x, y = load_dataset(dataset)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=0)
    cb = CatBoostClassifier(iterations=30, depth=4, verbose=0, random_seed=0, thread_count=4,
                            allow_writing_files=False).fit(x_tr, y_tr)
    background = x_tr[np.random.default_rng(0).choice(len(x_tr), size=100, replace=False)]
    return cb, background, x_tr, x_te, y_te


def clear() -> list[dict[str, object]]:
    rows = []
    for dataset in DATASETS:
        cb, background, x_tr, x_te, y_te = setup(dataset)
        raw = cb.predict(x_te, prediction_type="RawFormulaVal")
        for bits in BITS:
            model = cipherforest.from_catboost(cb, background=background, x_train=x_tr, output_bits=bits)
            score, _ = cipherforest.explain(model, x_te)
            rows.append({"dataset": dataset, "output_bits": bits, "auc_float": float(roc_auc_score(y_te, raw)),
                         "auc": float(roc_auc_score(y_te, score)), "bound": len(model.trees) / (2 * model.scale),
                         "max_error_logodds": float(np.abs(score - raw).max()),
                         "max_error_probability": float(np.abs(sigmoid(score) - sigmoid(raw)).max()),
                         "mean_error_probability": float(np.abs(sigmoid(score) - sigmoid(raw)).mean())})
    return rows


def timed(bits: int, output: str, requests: int = 20) -> dict[str, object]:
    from cipherforest.circuit import compile_model
    from cipherforest.deploy import CLIENT_FILE, SERVER_FILE, Client, Server, export

    cb, background, x_tr, x_te, _ = setup("adult")
    model = cipherforest.from_catboost(cb, background=background, x_train=x_tr, output_bits=bits)
    work = Path("/tmp/precision")
    shutil.rmtree(work, ignore_errors=True)
    compiled = compile_model(model, output=output)
    export(compiled, model, work)
    server, client = Server(work / SERVER_FILE), Client.load(work / CLIENT_FILE)
    serialized = client.evaluation_keys()
    keys = server.load_keys(serialized)
    want, want_shap = cipherforest.explain(model, x_te[:requests])
    run_s = []
    for row, s, phi in zip(x_te[:requests], want, want_shap, strict=True):
        ciphertext = client.encrypt(row)
        start = time.perf_counter()
        result = server.run(ciphertext, keys)
        run_s.append(time.perf_counter() - start)
        got, got_shap = client.decrypt(result)
        assert abs(got - s) < 1e-9 and (output != "shap" or np.allclose(got_shap, phi)), "differs from explain"
    shutil.rmtree(work, ignore_errors=True)
    return {"dataset": "adult", "output_bits": bits, "output": output, "requests": requests,
            "pbs_count": compiled.circuit.programmable_bootstrap_count, "complexity": compiled.circuit.complexity,
            "evaluation_keys_mb": len(serialized) / 2**20,
            "server_run_s_median": float(np.median(run_s)), "server_run_s_max": float(np.max(run_s))}


if __name__ == "__main__":
    if "--time" in sys.argv:
        i = sys.argv.index("--time")
        row = timed(int(sys.argv[i + 1]), sys.argv[i + 2])
        rows = json.loads(TIMES.read_text()) if TIMES.exists() else []
        TIMES.write_text(json.dumps([*rows, row], indent=1))
        print(json.dumps(row), flush=True)
    else:
        rows = clear()
        OUT.write_text(json.dumps(rows, indent=1))
        for r in rows:
            print(r["dataset"], r["output_bits"], round(r["auc"], 4), round(r["max_error_logodds"], 3),
                  round(r["max_error_probability"], 3), round(r["mean_error_probability"], 4), flush=True)
