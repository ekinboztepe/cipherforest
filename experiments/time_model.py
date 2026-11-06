"""Server time of one exported model, 20 requests after keygen. Docker image, one output per process.

    python experiments/time_model.py <model.json> <rows.npy> <shap|score|decision> [threshold]

Every answer is checked against the integer reference.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from cipherforest.circuit import compile_model
from cipherforest.deploy import CLIENT_FILE, SERVER_FILE, Client, Server, export
from cipherforest.quantize import QuantizedModel

if __name__ == "__main__":
    model = QuantizedModel.from_json(Path(sys.argv[1]).read_text())
    rows, output = np.load(sys.argv[2])[:20], sys.argv[3]
    threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    cut = model.cut_for(threshold) if output == "decision" else None
    work = Path("/tmp/time_model")
    shutil.rmtree(work, ignore_errors=True)
    compiled = compile_model(model, output=output, cut=cut)
    export(compiled, model, work)
    server, client = Server(work / SERVER_FILE), Client.load(work / CLIENT_FILE)
    keys = server.load_keys(client.evaluation_keys())
    pred, _ = model.decode(model.simulate(model.quantizer.transform(rows)))
    times = []
    for row, p in zip(rows, pred, strict=True):
        ciphertext = client.encrypt(row)
        start = time.perf_counter()
        result = server.run(ciphertext, keys)
        times.append(time.perf_counter() - start)
        got, _ = client.decrypt(result)
        want = float(p > threshold) if output == "decision" else p
        assert abs(got - want) < 1e-9, "encrypted result differs from the integer reference"
    print(json.dumps({"pbs": compiled.circuit.programmable_bootstrap_count, "complexity": compiled.circuit.complexity,
                      "server_run_s_median": float(np.median(times)), "server_run_s_max": float(max(times))}))
