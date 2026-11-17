"""Multiclass CatBoost served encrypted: one pass for every class.

Covertype (7 forest cover types, 54 features), 20000 rows sampled with seed 0, a
quarter held out. CatBoost MultiClass, depth 4 x 30. A multiclass model shares every
split across classes, so the circuit compares once per tree and looks up once per
class (MultiClassModel). In the clear: accuracy of CatBoost and of the served model
(argmax of the class scores), and the largest probability error after softmax.
With --time OUTPUT (Docker, docker/Dockerfile plus catboost and scikit-learn): 10
encrypted requests through the exported files, every answer equal to explain; and the
same for one class alone, which is what serving each class as its own model would
cost per class.

    uv run python experiments/multiclass_bench.py
    python experiments/multiclass_bench.py --time shap
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

import cipherforest

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "multiclass_bench.json"


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def setup():
    data = fetch_covtype(data_home=str(HERE / ".data"))
    rows = np.random.default_rng(0).choice(len(data.target), size=20_000, replace=False)
    x, y = data.data[rows].astype(np.float64), data.target[rows] - 1
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=0)
    cb = CatBoostClassifier(iterations=30, depth=4, loss_function="MultiClass", verbose=0, random_seed=0,
                            thread_count=4, allow_writing_files=False).fit(x_tr, y_tr)
    background = x_tr[np.random.default_rng(0).choice(len(x_tr), size=100, replace=False)]
    return cb, cipherforest.from_catboost(cb, background=background, x_train=x_tr), x_te, y_te


def clear() -> dict[str, object]:
    cb, model, x_te, y_te = setup()
    raw = cb.predict(x_te, prediction_type="RawFormulaVal")
    scores, _ = cipherforest.explain(model, x_te)
    return {"classes": len(model.classes), "accuracy_catboost": float(np.mean(raw.argmax(axis=1) == y_te)),
            "accuracy_served": float(np.mean(scores.argmax(axis=1) == y_te)),
            "same_class_as_catboost": float(np.mean(scores.argmax(axis=1) == raw.argmax(axis=1))),
            "max_probability_error": float(np.abs(softmax(scores) - softmax(raw)).max()),
            "pbs_shared": int(sum(t.features.size + len(model.classes) * t.players.size for t in model.trees)),
            "pbs_one_model_per_class": int(len(model.classes) * sum(t.features.size + t.players.size
                                                                   for t in model.trees))}


def timed(output: str, requests: int = 10) -> list[dict[str, object]]:
    from cipherforest.circuit import compile_model
    from cipherforest.deploy import CLIENT_FILE, SERVER_FILE, Client, Server, export

    _, model, x_te, _ = setup()
    rows = []
    for name, served in (("all classes", model), ("one class", model.classes[0])):
        work = Path("/tmp/multiclass")
        shutil.rmtree(work, ignore_errors=True)
        compiled = compile_model(served, output=output)
        export(compiled, served, work)
        server, client = Server(work / SERVER_FILE), Client.load(work / CLIENT_FILE)
        serialized = client.evaluation_keys()
        keys = server.load_keys(serialized)
        want, _ = cipherforest.explain(served, x_te[:requests])
        run_s = []
        for row, w in zip(x_te[:requests], want, strict=True):
            ciphertext = client.encrypt(row)
            start = time.perf_counter()
            result = server.run(ciphertext, keys)
            run_s.append(time.perf_counter() - start)
            got, _ = client.decrypt(result)
            assert np.allclose(got, w, atol=1e-9), "encrypted scores differ from explain"
        rows.append({"served": name, "output": output, "requests": requests,
                     "pbs_count": compiled.circuit.programmable_bootstrap_count,
                     "complexity": compiled.circuit.complexity, "evaluation_keys_mb": len(serialized) / 2**20,
                     "server_run_s_median": float(np.median(run_s))})
        print(json.dumps(rows[-1]), flush=True)
    return rows


if __name__ == "__main__":
    found = json.loads(OUT.read_text()) if OUT.exists() else {}
    if "--time" in sys.argv:
        output = sys.argv[sys.argv.index("--time") + 1]
        found.setdefault("encrypted", [])
        found["encrypted"] += timed(output)
    else:
        found["clear"] = clear()
        print(json.dumps(found["clear"]))
    OUT.write_text(json.dumps(found, indent=1))
