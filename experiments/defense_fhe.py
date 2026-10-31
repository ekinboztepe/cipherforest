"""What the defences cost under FHE, part 2 (Linux/Docker).

Variants of the e2e model, one per process to keep memory down:

- base: model.json as deployed.
- noise-b: the same circuit plus clear rounded Laplace(b) from the server,
  clipped to +-8b.
- grid-p, bits-k: from defense_fhe_prepare.py.

Each one is compiled, deployed and run on the e2e samples through the
exported files. Every decrypted result must equal the integer reference plus
the noise the server drew. Results go to e2e/<dataset>/defense_fhe.json.

Wide grids need keys of several GB. --compile-only stops before keygen and
records Concrete's key sizes and costs, which needs little memory.

    python experiments/defense_fhe.py <dataset> <variant> [--compile-only]
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from cipherforest.circuit import compile_model
from cipherforest.deploy import CLIENT_FILE, SERVER_FILE, Client, Server, export, laplace_noise
from cipherforest.quantize import QuantizedModel

E2E = Path(__file__).resolve().parent / "e2e"
N_SAMPLES = 10


def load(directory: Path, variant: str) -> tuple[QuantizedModel, float]:
    if variant == "base":
        return QuantizedModel.from_json((directory / "model.json").read_text()), 0.0
    if variant.startswith("noise-"):
        return QuantizedModel.from_json((directory / "model.json").read_text()), float(variant.removeprefix("noise-"))
    return QuantizedModel.from_json((directory / "variants" / f"{variant}.json").read_text()), 0.0


def run(directory: Path, variant: str, keygen: bool) -> dict[str, object]:
    model, b = load(directory, variant)
    bound = math.ceil(8 * b)
    x = np.load(directory / "samples.npz")["x"][:N_SAMPLES]

    start = time.perf_counter()
    compiled = compile_model(model, noise_bound=bound)
    compile_s = time.perf_counter() - start
    circuit = compiled.circuit
    static = {
        "noise_b": b,
        "noise_bound": bound,
        "input_bits": model.quantizer.input_bits,
        "output_bits": model.output_bits,
        "max_bit_width": circuit.graph.maximum_integer_bit_width(),
        "pbs_count": circuit.programmable_bootstrap_count,
        "complexity": circuit.complexity,
        "compile_s": compile_s,
        "bootstrap_keys_mb": circuit.size_of_bootstrap_keys / 2**20,
        "keyswitch_keys_mb": circuit.size_of_keyswitch_keys / 2**20,
        "ciphertext_kb": circuit.size_of_inputs / 2**10,
    }
    if not keygen:
        return static
    deploy = directory / "deploy" / variant
    export(compiled, model, deploy)

    server = Server(deploy / SERVER_FILE)
    start = time.perf_counter()
    client = Client.load(deploy / CLIENT_FILE)
    keygen_s = time.perf_counter() - start
    serialized_keys = client.evaluation_keys()
    keys_mb = len(serialized_keys) / 2**20
    keys = server.load_keys(serialized_keys)
    del serialized_keys

    want = model.simulate(model.quantizer.transform(x))[:, compiled.active_features]
    rng = np.random.default_rng(0)
    run_s = []
    for i, sample in enumerate(x):
        noise = laplace_noise(b, bound, compiled.active_features.size, rng) if bound else None
        ciphertext = client.encrypt(sample)
        t0 = time.perf_counter()
        result = server.run(ciphertext, keys, noise)
        run_s.append(time.perf_counter() - t0)
        _, shap = client.decrypt(result)
        got = np.rint(shap[compiled.active_features] * model.scale).astype(np.int64)
        if not np.array_equal(got, want[i] + (0 if noise is None else noise)):
            raise AssertionError(f"sample {i}: encrypted result differs from the integer reference")

    return static | {
        "samples": len(x),
        "all_match_integer_reference": True,
        "keygen_s": keygen_s,
        "evaluation_keys_mb": keys_mb,
        "server_run_s_median": float(np.median(run_s)),
        "server_run_s_max": float(np.max(run_s)),
    }


if __name__ == "__main__":
    dataset, variant = sys.argv[1:3]
    directory = E2E / dataset
    result = run(directory, variant, keygen="--compile-only" not in sys.argv)
    path = directory / "defense_fhe.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results[variant] = results.get(variant, {}) | result  # a compile-only run keeps the measured times
    path.write_text(json.dumps(results, indent=2))
    print(dataset, variant, json.dumps(result), flush=True)
