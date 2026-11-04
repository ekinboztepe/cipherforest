"""One encrypted measurement on the benchmark server, one per process.

Variants as in concrete_ml_bench.py: ours-T-default, ours-T-cml, xgb-T-D. The seed
changes the train/test split, the requests and the SHAP background; our teacher
forest keeps its fixed seed. Every finished job is appended to results/server_bench.json
right away, so a run cut short keeps what it measured. server_run_s holds every
request's time; P99 needs at least 100 of them.

    python experiments/server_bench.py <dataset> <seed> <variant> <requests>
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from concrete_ml_bench import WORK, ours, xgb
from datasets import load_dataset
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parent / "results" / "server_bench.json"


if __name__ == "__main__":
    dataset, seed, variant, requests = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    x, y = load_dataset(dataset)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=seed)
    x_run = x_te[np.random.default_rng(seed).choice(len(x_te), size=requests, replace=False)]
    kind, n_trees, rest = variant.split("-")

    start = time.perf_counter()
    if kind == "xgb":
        result = xgb(x_tr, y_tr, x_te, y_te, x_run, True, int(n_trees), int(rest), seed)
    else:
        result = ours(x_tr, y_tr, x_te, y_te, x_run, True, int(n_trees), rest, seed)
    row = {"dataset": dataset, "seed": seed, "variant": variant, "job_s": time.perf_counter() - start, **result}

    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    OUT.write_text(json.dumps([*rows, row], indent=1))
    shutil.rmtree(WORK, ignore_errors=True)
    print("RESULT", json.dumps(row), flush=True)  # the pod's log is how the rows leave it
