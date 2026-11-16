"""Can a retrained model keep the clients' keys?

Each client uploads hundreds of MB of evaluation keys, and a model retrained every
month would make every client do it again. compile_model(..., same_keys_as=old
client.json) compiles the new model within the old deployment's keyset (Concrete's
keyset restriction). This checks, per scenario, that the new circuit compiles that
way, that the server answers with the evaluation keys it already holds and the client
decrypts with the secret key it already has, every answer equal to explain, and what
the restriction costs against compiling the new model freely.

The old deployment is CatBoost depth 4 x 30 on Adult, trained on half of the
training split. Scenarios for the new model:

- retrain: same shape, other half of the data, other seed
- more-trees: 60 trees
- deeper: depth 6
- categorical: categories as CatBoost handles them instead of ordinal codes
- score: the retrained model with the prediction-only output

Strategy "default" is SAFE as shipped; "mono" adds Concrete's single-parameter
strategy to all three compilations, which might leave more room for a new model.

One scenario per process, in docker/Dockerfile (Concrete 2.11, as the package pins)
with catboost, scikit-learn and pandas added:

    python experiments/key_reuse_bench.py <scenario> [default|mono]
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from concrete import fhe
from datasets import load_dataset, load_raw
from sklearn.model_selection import train_test_split

import cipherforest
from cipherforest.circuit import SAFE, compile_model
from cipherforest.deploy import CLIENT_FILE, SERVER_FILE, Client, Server, export

OUT = Path(__file__).resolve().parent / "results" / "key_reuse_bench.json"
WORK = Path("/tmp/key_reuse")
REQUESTS = 10


def catboost_model(x, y, cat, trees: int, depth: int, seed: int) -> cipherforest.QuantizedModel:
    params = {"cat_features": cat, "max_ctr_complexity": 1} if cat else {}
    cb = CatBoostClassifier(iterations=trees, depth=depth, verbose=0, random_seed=seed, thread_count=4,
                            allow_writing_files=False, **params).fit(x.tolist() if cat else x, y)
    background = x[np.random.default_rng(seed).choice(len(x), size=100, replace=False)]
    return cipherforest.from_catboost(cb, background=background, x_train=x)


def timed(server: Server, client: Client, keys: object, model, rows, output: str) -> list[float]:
    want_score, want_shap = cipherforest.explain(model, rows)
    run_s = []
    for row, s, phi in zip(rows, want_score, want_shap, strict=True):
        ciphertext = client.encrypt(row)
        start = time.perf_counter()
        result = server.run(ciphertext, keys)
        run_s.append(time.perf_counter() - start)
        got, got_shap = client.decrypt(result)
        assert abs(got - s) < 1e-9, "encrypted score differs from explain"
        if output == "shap":
            assert np.allclose(got_shap, phi), "encrypted SHAP differs from explain"
    return run_s


def run(scenario: str, strategy: str) -> dict[str, object]:
    cfg = SAFE if strategy == "default" else SAFE.fork(parameter_selection_strategy=fhe.ParameterSelectionStrategy.MONO)
    x, y = load_dataset("adult")
    x_raw, cat, _ = load_raw("adult")
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=0)
    first, second = tr[: len(tr) // 2], tr[len(tr) // 2:]

    shutil.rmtree(WORK, ignore_errors=True)
    old = catboost_model(x[first], y[first], [], 30, 4, 0)
    export(compile_model(old, configuration=cfg), old, WORK / "old")
    old_client = Client.load(WORK / "old" / CLIENT_FILE)
    evaluation_keys, secret_keys = old_client.evaluation_keys(), old_client.secret_keys()

    output = "score" if scenario == "score" else "shap"
    trees, depth = {"more-trees": (60, 4), "deeper": (30, 6)}.get(scenario, (30, 4))
    if scenario == "categorical":
        new, rows = catboost_model(x_raw[second], y[second], cat, trees, depth, 1), x_raw[te[:REQUESTS]]
    else:
        new, rows = catboost_model(x[second], y[second], [], trees, depth, 1), x[te[:REQUESTS]]

    row: dict[str, object] = {"scenario": scenario, "strategy": strategy, "trees": trees, "depth": depth, "output": output,
                              "evaluation_keys_mb": len(evaluation_keys) / 2**20}
    start = time.perf_counter()
    try:
        restricted = compile_model(new, configuration=cfg, output=output, same_keys_as=WORK / "old" / CLIENT_FILE)
    except ValueError as error:
        return row | {"fits_old_keys": False, "error": str(error)[:300]}
    row["compile_restricted_s"] = time.perf_counter() - start
    export(restricted, new, WORK / "new")
    server = Server(WORK / "new" / SERVER_FILE)
    client = Client.load(WORK / "new" / CLIENT_FILE, keys=secret_keys)
    restricted_s = timed(server, client, server.load_keys(evaluation_keys), new, rows, output)

    free = compile_model(new, configuration=cfg, output=output)
    export(free, new, WORK / "free")
    server = Server(WORK / "free" / SERVER_FILE)
    client = Client.load(WORK / "free" / CLIENT_FILE)
    free_keys = client.evaluation_keys()
    free_s = timed(server, client, server.load_keys(free_keys), new, rows, output)
    return row | {
        "fits_old_keys": True,
        "complexity_restricted": restricted.circuit.complexity, "complexity_free": free.circuit.complexity,
        "server_run_s_restricted": float(np.median(restricted_s)), "server_run_s_free": float(np.median(free_s)),
        "free_evaluation_keys_mb": len(free_keys) / 2**20, "requests": REQUESTS,
    }


if __name__ == "__main__":
    result = run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "default")
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    OUT.write_text(json.dumps([*rows, result], indent=1))
    shutil.rmtree(WORK, ignore_errors=True)
    print(json.dumps(result), flush=True)
