"""Concrete-ML's own trees against ours, same data, same Concrete, same laptop.

Variants, one per process (docker/concrete-ml.Dockerfile, which pins
concrete-python 2.10):

- xgb-T-D: Concrete-ML XGBClassifier, T trees of depth D, n_bits=6. Prediction
  only.
- ours-T-default, ours-T-cml: the e2e pipeline (forest teacher, T oblivious
  trees of depth 6, 8-bit outputs) trained here. Prediction and exact SHAP.
  default compiles with Concrete's defaults as in the e2e demo; cml with
  Concrete-ML's settings (compressed keys and inputs, p_error 2^-40 per
  bootstrap), for a like-for-like comparison.

Both run through their own client/server files on 10 test rows. Keys take
several GB for the larger models and the laptop has crashed on less, so
--compile-only records Concrete's estimates without keygen, and --samples runs
fewer rows of a slow model. Results go to results/concrete_ml_bench.json.

    python experiments/concrete_ml_bench.py <dataset> <variant> [--compile-only] [--samples N]
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from concrete import fhe
from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import XGBClassifier
from datasets import load_dataset
from defense_fhe import measure
from e2e_prepare import N_BINS, OUTPUT_BITS, SEED, student
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cipherforest.quantize import FeatureQuantizer, quantize_model

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "concrete_ml_bench.json"
WORK = Path("/tmp/concrete_ml_bench")
N_SAMPLES = 10
LIKE_CONCRETE_ML = fhe.Configuration(compress_evaluation_keys=True, compress_input_ciphertexts=True, p_error=2**-40)


def circuit_sizes(circuit) -> dict[str, float]:
    return {
        "pbs_count": circuit.programmable_bootstrap_count,
        "max_bit_width": circuit.graph.maximum_integer_bit_width(),
        "complexity": circuit.complexity,
        "bootstrap_keys_mb": circuit.size_of_bootstrap_keys / 2**20,
        "keyswitch_keys_mb": circuit.size_of_keyswitch_keys / 2**20,
        "ciphertext_kb": circuit.size_of_inputs / 2**10,
    }


def xgb(x_tr, y_tr, x_te, y_te, x_run, keygen: bool, n_trees: int, depth: int) -> dict[str, object]:
    model = XGBClassifier(n_bits=6, n_estimators=n_trees, max_depth=depth, random_state=SEED, n_jobs=1)
    model.fit(x_tr, y_tr)
    start = time.perf_counter()
    model.compile(x_tr)
    row: dict[str, object] = {
        "compile_s": time.perf_counter() - start,
        "auc": float(roc_auc_score(y_te, model.predict_proba(x_te, fhe="simulate")[:, 1])),
        **circuit_sizes(model.fhe_circuit),
    }
    if not keygen:
        return row

    shutil.rmtree(WORK, ignore_errors=True)
    FHEModelDev(str(WORK / "model"), model).save()
    server = FHEModelServer(str(WORK / "model"))
    server.load()
    client = FHEModelClient(str(WORK / "model"), key_dir=str(WORK / "keys"))
    start = time.perf_counter()
    client.generate_private_and_evaluation_keys()
    keygen_s = time.perf_counter() - start
    keys = client.get_serialized_evaluation_keys()

    want = model.predict_proba(x_run, fhe="simulate")
    run_s, diff = [], 0.0
    for i, sample in enumerate(x_run):
        ciphertext = client.quantize_encrypt_serialize(sample[None])
        t0 = time.perf_counter()
        result = server.run(ciphertext, keys)
        run_s.append(time.perf_counter() - t0)
        print(f"sample {i}: {run_s[-1]:.1f} s", flush=True)
        diff = max(diff, float(np.abs(client.deserialize_decrypt_dequantize(result) - want[i]).max()))
    return row | {
        "samples": len(x_run),
        "max_diff_vs_simulate": diff,
        "keygen_s": keygen_s,
        "evaluation_keys_mb": len(keys) / 2**20,
        "server_run_s_median": float(np.median(run_s)),
        "server_run_s_max": float(np.max(run_s)),
    }


def ours(x_tr, y_tr, x_te, y_te, x_run, keygen: bool, n_trees: int, settings: str) -> dict[str, object]:
    quantizer = FeatureQuantizer.fit(x_tr, N_BINS)
    teacher, ensemble, tables = student(x_tr, y_tr, quantizer, np.random.default_rng(SEED), n_trees)
    model = quantize_model(ensemble, tables, quantizer, OUTPUT_BITS)
    pred, _ = model.decode(model.simulate(quantizer.transform(x_te)))
    row: dict[str, object] = {
        "auc_teacher": float(roc_auc_score(y_te, teacher.predict_proba(x_te)[:, 1])),
        "auc_float": float(roc_auc_score(y_te, ensemble.predict(x_te))),
        "auc": float(roc_auc_score(y_te, pred)),
    }
    configuration = {"default": None, "cml": LIKE_CONCRETE_ML}[settings]
    return row | measure(model, 0.0, x_run, WORK, keygen, configuration)


if __name__ == "__main__":
    dataset, variant = sys.argv[1:3]
    x, y = load_dataset(dataset)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y, random_state=SEED)
    x_run = x_te[np.random.default_rng(SEED).choice(len(x_te), size=N_SAMPLES, replace=False)]
    if "--samples" in sys.argv:
        x_run = x_run[: int(sys.argv[sys.argv.index("--samples") + 1])]
    kind, n_trees, rest = variant.split("-")
    keygen = "--compile-only" not in sys.argv
    if kind == "xgb":
        result = xgb(x_tr, y_tr, x_te, y_te, x_run, keygen, int(n_trees), int(rest))
    else:
        result = ours(x_tr, y_tr, x_te, y_te, x_run, keygen, int(n_trees), rest)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    runs = results.setdefault(dataset, {})
    runs[variant] = runs.get(variant, {}) | result  # a compile-only run keeps the measured times
    OUT.write_text(json.dumps(results, indent=2))
    print(dataset, variant, json.dumps(result), flush=True)
