"""How many queries a client needs to find every threshold of the e2e models.

Needs experiments/e2e/<dataset>/model.json from e2e_prepare.py.

    uv run python experiments/extraction_bench.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cipherforest.extraction import CountingOracle, find_thresholds, plaintext_oracle, recall, true_thresholds
from cipherforest.quantize import QuantizedModel

E2E = Path(__file__).resolve().parent / "e2e"
OUT = Path(__file__).resolve().parent / "results" / "extraction_bench.json"
DATASETS = ("adult", "credit_default")
STARTS = (1, 2, 3, 5)
SEEDS = range(5)


def run(name: str) -> list[dict[str, object]]:
    model = QuantizedModel.from_json((E2E / name / "model.json").read_text())
    truth = true_thresholds(model)
    rows = []
    for use_shap in (True, False):
        for n_starts in STARTS:
            for seed in SEEDS:
                oracle = CountingOracle(plaintext_oracle(model))
                found = find_thresholds(oracle, model.quantizer, n_starts, np.random.default_rng(seed), use_shap)
                rows.append(
                    {
                        "dataset": name,
                        "oracle": "pred+shap" if use_shap else "pred",
                        "starts": n_starts,
                        "seed": seed,
                        "queries": oracle.queries,
                        "thresholds": len(truth),
                        "recall": recall(found, truth),
                        "false_positives": len(found - truth),
                    }
                )
    return rows


if __name__ == "__main__":
    rows = [r for name in DATASETS for r in run(name)]
    OUT.write_text(json.dumps(rows, indent=1))
    print("dataset         oracle     starts  queries  thresholds  recall (min / mean over seeds)")
    for name in DATASETS:
        for oracle in ("pred+shap", "pred"):
            for n_starts in STARTS:
                sel = [r for r in rows if (r["dataset"], r["oracle"], r["starts"]) == (name, oracle, n_starts)]
                rec = [float(r["recall"]) for r in sel]
                print(
                    f"{name:15} {oracle:10} {n_starts:6} {sel[0]['queries']:8} {sel[0]['thresholds']:11}"
                    f"  {min(rec):.2f} / {np.mean(rec):.2f}"
                )
