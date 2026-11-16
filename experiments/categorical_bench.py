"""Categories as categories: CatBoost's own handling against ordinal codes, in the clear.

Until now every dataset went through datasets._to_matrix, which turns a category
into its alphabetical position and fills missing numbers with the median, so
CatBoost saw "education" as a number. This trains CatBoost depth 4 x 30 trees both
ways on the datasets with categorical columns:

- ordinal: the matrix as before.
- categorical: the raw columns (datasets.load_raw), cat_features set, missing
  numbers left as NaN, max_ctr_complexity=1 so every split reads one feature.

AUC is that of the integer model the circuit serves (cipherforest.explain). The
served score must equal CatBoost's raw score up to the output rounding bound on the
whole test split, or the run fails. PBS is the SHAP circuit's bootstrap count.

    uv run python experiments/categorical_bench.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool
from datasets import load_dataset, load_raw
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import cipherforest

OUT = Path(__file__).resolve().parent / "results" / "categorical_bench.json"
DATASETS = ("adult", "bank_marketing", "credit_g")
SEEDS = (0, 1, 2)
TREES, DEPTH = 30, 4


def pbs(model: cipherforest.QuantizedModel) -> int:
    return int(sum(t.features.size + t.players.size for t in model.trees))


def run(dataset: str, seed: int) -> list[dict[str, object]]:
    x_raw, cat, y = load_raw(dataset)
    x_ord, _ = load_dataset(dataset)
    rows = []
    for kind, x, cat_features in (("ordinal", x_ord, []), ("categorical", x_raw, cat)):
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=seed)
        background = x_tr[np.random.default_rng(seed).choice(len(x_tr), size=100, replace=False)]
        params = {"max_ctr_complexity": 1} if cat_features else {}
        cb = CatBoostClassifier(iterations=TREES, depth=DEPTH, verbose=0, random_seed=seed, thread_count=4,
                                cat_features=cat_features or None, allow_writing_files=False, **params)
        cb.fit(x_tr.tolist() if cat_features else x_tr, y_tr)
        start = time.perf_counter()
        model = cipherforest.from_catboost(cb, background=background, x_train=x_tr)
        import_s = time.perf_counter() - start
        score, _ = cipherforest.explain(model, x_te)
        raw = cb.predict(Pool(x_te.tolist(), cat_features=cat_features) if cat_features else x_te,
                         prediction_type="RawFormulaVal")
        bound = len(model.trees) / (2 * model.scale)
        error = float(np.abs(score - raw).max())
        if error > bound + 1e-9:
            raise AssertionError(f"{dataset} {kind}: served score off by {error}, bound {bound}")
        categorical_levels = sum(s is not None for t in model.trees for s in t.sets)
        rows.append({
            "dataset": dataset, "seed": seed, "kind": kind,
            "auc_float": float(roc_auc_score(y_te, raw)), "auc": float(roc_auc_score(y_te, score)),
            "max_error": error, "bound": bound, "pbs": pbs(model), "input_bits": model.quantizer.input_bits,
            "categorical_levels": categorical_levels, "levels": sum(t.features.size for t in model.trees),
            "import_s": import_s,
        })
    return rows


def timed(dataset: str, kind: str, requests: int = 20) -> dict[str, object]:
    """Seed 0, encrypted through the exported files on SAFE; every answer checked against explain."""
    import shutil

    from cipherforest.circuit import SAFE, compile_model
    from cipherforest.deploy import CLIENT_FILE, SERVER_FILE, Client, Server, export

    x_raw, cat, y = load_raw(dataset)
    x = x_raw if kind == "categorical" else load_dataset(dataset)[0]
    cat_features = cat if kind == "categorical" else []
    x_tr, x_te, y_tr, _ = train_test_split(x, y, test_size=0.25, stratify=y, random_state=0)
    params = {"max_ctr_complexity": 1} if cat_features else {}
    cb = CatBoostClassifier(iterations=TREES, depth=DEPTH, verbose=0, random_seed=0, thread_count=4,
                            cat_features=cat_features or None, allow_writing_files=False, **params)
    cb.fit(x_tr.tolist() if cat_features else x_tr, y_tr)
    background = x_tr[np.random.default_rng(0).choice(len(x_tr), size=100, replace=False)]
    model = cipherforest.from_catboost(cb, background=background, x_train=x_tr)
    work = Path("/tmp/categorical_time")
    shutil.rmtree(work, ignore_errors=True)
    export(compile_model(model, configuration=SAFE), model, work)
    server, client = Server(work / SERVER_FILE), Client.load(work / CLIENT_FILE)
    keys = server.load_keys(client.evaluation_keys())
    want_score, want_shap = cipherforest.explain(model, x_te[:requests])
    run_s = []
    for row, s, phi in zip(x_te[:requests], want_score, want_shap, strict=True):
        ciphertext = client.encrypt(row)
        start = time.perf_counter()
        result = server.run(ciphertext, keys)
        run_s.append(time.perf_counter() - start)
        got, got_shap = client.decrypt(result)
        assert abs(got - s) < 1e-9 and np.allclose(got_shap, phi), "encrypted result differs from explain"
    return {"dataset": dataset, "kind": kind, "requests": requests, "pbs": pbs(model),
            "server_run_s_median": float(np.median(run_s)), "server_run_s_max": float(np.max(run_s))}


if __name__ == "__main__" and "--time" in sys.argv:
    # Docker (docker/concrete-ml.Dockerfile), one kind per process: --time <dataset> <kind>
    _, dataset, kind = sys.argv[sys.argv.index("--time"):][:3]
    times = OUT.with_name("categorical_time.json")
    found = json.loads(times.read_text()) if times.exists() else []
    found.append(timed(dataset, kind))
    times.write_text(json.dumps(found, indent=1))
    print(found[-1], flush=True)
elif __name__ == "__main__":
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    for dataset in sys.argv[1:] or DATASETS:
        for seed in SEEDS:
            if any(r["dataset"] == dataset and r["seed"] == seed for r in rows):
                continue
            new = run(dataset, seed)
            rows += new
            OUT.write_text(json.dumps(rows, indent=1))
            for r in new:
                print(dataset, seed, r["kind"], round(r["auc"], 4), r["pbs"], r["input_bits"],
                      f"{r['categorical_levels']}/{r['levels']}", f"err {r['max_error']:.3g} <= {r['bound']:.3g}",
                      flush=True)
