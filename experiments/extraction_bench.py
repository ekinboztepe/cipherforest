"""How many queries a client needs to copy the e2e models exactly.

The attacker adds one random starting point per round until its copy agrees with
the server on 200 fresh random queries; those checks count towards the total.
Needs experiments/e2e/<dataset>/model.json from e2e_prepare.py.

    uv run python experiments/extraction_bench.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cipherforest.extraction import (
    CountingOracle,
    code_values,
    extract,
    plaintext_oracle,
    prediction_interactions,
    prediction_only,
    shap_interactions,
)
from cipherforest.quantize import QuantizedModel

E2E = Path(__file__).resolve().parent / "e2e"
OUT = Path(__file__).resolve().parent / "results" / "extraction_bench.json"
DATASETS = ("adult", "credit_default")
SEEDS = range(10)
N_EVAL = 5000

# attacker: (sees SHAP, how it finds the interaction graph)
ATTACKERS = {
    "pred, full grid": (False, None),
    "pred, graph": (False, prediction_interactions),
    "pred+shap, graph": (True, shap_interactions),
}


def attack(model: QuantizedModel, name: str, seed: int) -> dict[str, object]:
    sees_shap, graph = ATTACKERS[name]
    view = plaintext_oracle(model) if sees_shap else prediction_only(plaintext_oracle(model))
    oracle = CountingOracle(view)
    copy, rounds = extract(oracle, model.quantizer, np.random.default_rng(seed), graph)

    rng = np.random.default_rng(1000 + seed)
    x = np.column_stack([rng.choice(code_values(e), size=N_EVAL) for e in model.quantizer.edges])
    got, want = np.column_stack(copy(x)), np.column_stack(view(x))
    return {
        "attacker": name,
        "seed": seed,
        "rounds": rounds,
        "queries": oracle.queries,
        "exact_rows": float(np.mean(np.all(np.abs(got - want) < 1e-9, axis=1))),
    }


def run(dataset: str) -> list[dict[str, object]]:
    model = QuantizedModel.from_json((E2E / dataset / "model.json").read_text())
    return [{"dataset": dataset, **attack(model, name, seed)} for name in ATTACKERS for seed in SEEDS]


if __name__ == "__main__":
    rows = [r for dataset in DATASETS for r in run(dataset)]
    OUT.write_text(json.dumps(rows, indent=1))
    print("dataset         attacker           queries (median, max)  rounds (max)  exact rows (min)")
    for dataset in DATASETS:
        for name in ATTACKERS:
            sel = [r for r in rows if r["dataset"] == dataset and r["attacker"] == name]
            q = [int(r["queries"]) for r in sel]
            print(
                f"{dataset:15} {name:18} {int(np.median(q)):8} {max(q):8}"
                f"  {max(int(r['rounds']) for r in sel):12}  {min(float(r['exact_rows']) for r in sel):.4f}"
            )
