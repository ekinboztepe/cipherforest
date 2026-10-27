"""End-to-end demo, part 2 (Linux/Docker): compile, deploy, run encrypted, verify, time.

Client and server only touch the exported files and exchange bytes. Every
decrypted result must equal the integer reference; fhe.json gets timings, sizes
and the error against the float student.

    python experiments/e2e_fhe.py [dataset ...]   # one dataset per process keeps memory down
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from cipherforest.circuit import compile_model
from cipherforest.deploy import CLIENT_FILE, SERVER_FILE, Client, Server, export
from cipherforest.quantize import QuantizedModel

ROOT = Path(__file__).resolve().parent / "e2e"


def run(directory: Path) -> dict[str, object]:
    model = QuantizedModel.from_json((directory / "model.json").read_text())
    samples = np.load(directory / "samples.npz")

    start = time.perf_counter()
    compiled = compile_model(model)
    compile_s = time.perf_counter() - start
    deploy = directory / "deploy"
    export(compiled, model, deploy)

    server = Server(deploy / SERVER_FILE)
    start = time.perf_counter()
    client = Client.load(deploy / CLIENT_FILE)
    keygen_s = time.perf_counter() - start
    serialized_keys = client.evaluation_keys()
    keys_mb = len(serialized_keys) / 2**20
    keys = server.load_keys(serialized_keys)
    del serialized_keys

    want_pred, want_shap = model.decode(model.simulate(model.quantizer.transform(samples["x"])))
    enc_s, run_s, dec_s, pred_err, shap_err = [], [], [], [], []
    ct_bytes = result_bytes = 0
    for i, x in enumerate(samples["x"]):
        t0 = time.perf_counter()
        ciphertext = client.encrypt(x)
        t1 = time.perf_counter()
        result = server.run(ciphertext, keys)
        t2 = time.perf_counter()
        pred, shap = client.decrypt(result)
        t3 = time.perf_counter()
        if not (np.array_equal(shap, want_shap[i]) and pred == want_pred[i]):
            raise AssertionError(f"sample {i}: encrypted result differs from the integer reference")
        enc_s.append(t1 - t0)
        run_s.append(t2 - t1)
        dec_s.append(t3 - t2)
        pred_err.append(abs(pred - samples["student_pred"][i]))
        shap_err.append(float(np.abs(shap - samples["student_shap"][i]).max()))
        ct_bytes, result_bytes = len(ciphertext), len(result)

    return {
        "samples": len(samples["x"]),
        "all_match_integer_reference": True,
        "pbs_count": compiled.circuit.programmable_bootstrap_count,
        "compile_s": compile_s,
        "keygen_s": keygen_s,
        "evaluation_keys_mb": keys_mb,
        "ciphertext_kb": ct_bytes / 2**10,
        "result_kb": result_bytes / 2**10,
        "server_zip_kb": (deploy / SERVER_FILE).stat().st_size / 2**10,
        "encrypt_s_median": float(np.median(enc_s)),
        "server_run_s_median": float(np.median(run_s)),
        "server_run_s_max": float(np.max(run_s)),
        "decrypt_s_median": float(np.median(dec_s)),
        "max_pred_error_vs_float_student": float(np.max(pred_err)),
        "max_shap_error_vs_float_student": float(np.max(shap_err)),
    }


if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for directory in sorted(p for p in ROOT.iterdir() if (p / "model.json").exists()):
        if wanted and directory.name not in wanted:
            continue
        result = run(directory)
        (directory / "fhe.json").write_text(json.dumps(result, indent=2))
        print(directory.name, json.dumps(result, indent=2), flush=True)
