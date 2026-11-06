"""What each output costs, against Concrete-ML's XGBoost on the same data. Compile only.

Our three outputs (shap, score, decision) on the e2e student, all on Concrete-ML's
settings (cipherforest.SAFE), and Concrete-ML's XGBClassifier, which returns the
prediction only. No keys are generated: Concrete's complexity estimate stands in
for server time, which it tracked with correlation 0.99 in server_bench (about
0.35 s per 1e10 for ours and 0.30 s for XGB on the EPYC 9655P).

Runs in docker/concrete-ml.Dockerfile, one dataset per process:

    python experiments/output_bench.py <dataset>

With --time T OUTPUT it instead runs N_TIMED requests through the exported files of
our T-tree model with that output, after keygen, and appends the times to
results/output_time.json. One output per process: the keys take over a GB.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from concrete_ml_bench import circuit_sizes, xgb
from datasets import load_dataset
from e2e_prepare import N_BINS, OUTPUT_BITS, SEED, student
from sklearn.model_selection import train_test_split

from cipherforest.circuit import SAFE, compile_model
from cipherforest.quantize import FeatureQuantizer, quantize_model

OUT = Path(__file__).resolve().parent / "results" / "output_bench.json"
TIMES = OUT.with_name("output_time.json")
TREES = (3, 10, 30)
N_TIMED = 20


def timed(dataset: str, x_tr, y_tr, x_te, n_trees: int, output: str) -> dict[str, object]:
    import shutil
    import time

    from cipherforest.deploy import CLIENT_FILE, SERVER_FILE, Client, Server, export

    quantizer = FeatureQuantizer.fit(x_tr, N_BINS)
    _, ensemble, tables = student(x_tr, y_tr, quantizer, np.random.default_rng(SEED), n_trees)
    model = quantize_model(ensemble, tables, quantizer, OUTPUT_BITS)
    cut = model.cut_for(0.5) if output == "decision" else None
    work = Path("/tmp/output_time")
    shutil.rmtree(work, ignore_errors=True)
    export(compile_model(model, configuration=SAFE, output=output, cut=cut), model, work)
    server, client = Server(work / SERVER_FILE), Client.load(work / CLIENT_FILE)
    keys = server.load_keys(client.evaluation_keys())
    pred, _ = model.decode(model.simulate(quantizer.transform(x_te[:N_TIMED])))
    run_s = []
    for x, p in zip(x_te[:N_TIMED], pred, strict=True):
        ciphertext = client.encrypt(x)
        start = time.perf_counter()
        result = server.run(ciphertext, keys)
        run_s.append(time.perf_counter() - start)
        got, _ = client.decrypt(result)
        want = float(p > 0.5) if output == "decision" else p
        assert abs(got - want) < 1e-9, "encrypted result differs from the integer reference"
    return {"dataset": dataset, "variant": f"ours-{n_trees}-{output}", "server_run_s_median": float(np.median(run_s)),
            "server_run_s_max": float(np.max(run_s)), "first_s": run_s[0], "requests": len(run_s)}


def ours(x_tr, y_tr, n_trees: int) -> list[dict[str, object]]:
    quantizer = FeatureQuantizer.fit(x_tr, N_BINS)
    _, ensemble, tables = student(x_tr, y_tr, quantizer, np.random.default_rng(SEED), n_trees)
    model = quantize_model(ensemble, tables, quantizer, OUTPUT_BITS)
    rows = []
    for output in ("shap", "score", "decision"):
        cut = model.cut_for(0.5) if output == "decision" else None
        circuit = compile_model(model, configuration=SAFE, output=output, cut=cut).circuit
        rows.append({"variant": f"ours-{n_trees}-{output}", **circuit_sizes(circuit)})
    return rows


if __name__ == "__main__":
    dataset = sys.argv[1]
    x, y = load_dataset(dataset)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=SEED)
    if "--time" in sys.argv:
        at = sys.argv.index("--time")
        row = timed(dataset, x_tr, y_tr, x_te, int(sys.argv[at + 1]), sys.argv[at + 2])
        print(json.dumps(row), flush=True)
        old = json.loads(TIMES.read_text()) if TIMES.exists() else []
        TIMES.write_text(json.dumps([r for r in old if (r["dataset"], r["variant"]) != (dataset, row["variant"])] + [row], indent=1))
        sys.exit(0)
    rows = []
    for n_trees in TREES:
        for row in ours(x_tr, y_tr, n_trees):
            rows.append({"dataset": dataset, **row})
            print(json.dumps(rows[-1]), flush=True)
        row = xgb(x_tr, y_tr, x_te, y_te, x_te[:1], False, n_trees, 6)
        rows.append({"dataset": dataset, "variant": f"xgb-{n_trees}-6", **{k: v for k, v in row.items() if k != "auc"}})
        print(json.dumps(rows[-1]), flush=True)
    old = json.loads(OUT.read_text()) if OUT.exists() else []
    OUT.write_text(json.dumps([r for r in old if r["dataset"] != dataset] + rows, indent=1))
