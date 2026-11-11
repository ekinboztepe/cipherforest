"""Do the copying attacks still work on the model the README recommends?

Every attack so far ran on the e2e student: 3 distilled trees of depth 6. The
recommended path is CatBoost served directly, depth 4 and 30 trees, which uses
more features and has more interactions. This runs the same two attacks on
CatBoost depth 4 at 10 and 30 trees, trained on the e2e split (random_state 0)
and imported with the same 32 training quantiles as edges:

- exact: extraction_bench's attackers, which rebuild the model from its
  thresholds and interaction graph. They give up (no copy) once the structure
  is past their limits (16 cliques, 2^17 cells per clique), which is a result too.
- label-only: label_bench's attacker against an approve/decline server, with
  and without 4 reasons on a decline, which learns a copy and needs no structure.
  The score is log-odds, so the server decides at 0 (probability 0.5).

    uv run python experiments/catboost_attack_bench.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from datasets import load_dataset
from extraction_bench import ATTACKERS, N_EVAL
from label_bench import DATASETS, REASONS
from label_bench import attack as label_attack
from sklearn.model_selection import train_test_split

import cipherforest
from cipherforest.extraction import Cached, Counted, code_values, extract, plaintext_oracle, prediction_only
from cipherforest.quantize import QuantizedModel

OUT = Path(__file__).resolve().parent / "results" / "catboost_attack_bench.json"
TREES = (10, 30)
DEPTH = 4
EXACT_SEEDS = range(5)
LABEL_SEEDS = range(3)
EXACT = ("pred, graph", "pred+shap, graph")


def model_for(dataset: str, n_trees: int) -> tuple[QuantizedModel, np.ndarray]:
    x, y = load_dataset(dataset)
    x_tr, x_te, y_tr, _ = train_test_split(x, y, test_size=0.25, stratify=y, random_state=0)
    background = x_tr[np.random.default_rng(0).choice(len(x_tr), size=100, replace=False)]
    cb = CatBoostClassifier(iterations=n_trees, depth=DEPTH, random_seed=0, thread_count=4, verbose=0,
                            allow_writing_files=False).fit(x_tr, y_tr)
    return cipherforest.from_catboost(cb, background=background, x_train=x_tr), x_te


def exact(model: QuantizedModel, name: str, seed: int) -> dict[str, object]:
    sees_shap, graph = ATTACKERS[name]
    view = plaintext_oracle(model) if sees_shap else prediction_only(plaintext_oracle(model))
    counter = Counted(view)
    copy, rounds = extract(Cached(counter), model.quantizer, np.random.default_rng(seed), graph)
    row: dict[str, object] = {"attack": "exact", "attacker": name, "seed": seed, "rounds": rounds,
                              "queries": counter.queries, "copied": copy is not None}
    if copy is not None:
        rng = np.random.default_rng(1000 + seed)
        x = np.column_stack([rng.choice(code_values(e), size=N_EVAL) for e in model.quantizer.edges])
        got, want = np.column_stack(copy(x)), np.column_stack(view(x))
        row["exact_rows"] = float(np.mean(np.all(np.abs(got - want) < 1e-9, axis=1)))
    return row


def done(rows: list[dict[str, object]], key: dict[str, object]) -> bool:
    return any(all(r.get(k) == v for k, v in key.items()) for r in rows)


def save(rows: list[dict[str, object]], row: dict[str, object]) -> None:
    rows.append(row)
    OUT.write_text(json.dumps(rows, indent=1))
    print({k: v for k, v in row.items() if k != "trace"}, flush=True)


def run(dataset: str, rows: list[dict[str, object]]) -> None:
    for n_trees in TREES:
        model, x_te = model_for(dataset, n_trees)
        base = {"dataset": dataset, "trees": n_trees, "depth": DEPTH,
                "features_used": len({int(f) for t in model.trees for f in t.features})}
        for name in EXACT:
            for seed in EXACT_SEEDS:
                if not done(rows, {"dataset": dataset, "trees": n_trees, "attacker": name, "seed": seed}):
                    start = time.perf_counter()
                    row = exact(model, name, seed)
                    save(rows, {**base, **row, "attack_s": time.perf_counter() - start})
        for reasons in (0, REASONS):
            for seed in LABEL_SEEDS:
                key = {"dataset": dataset, "trees": n_trees, "attack": "label", "reasons": reasons, "seed": seed}
                if not done(rows, key):
                    # CatBoost's score is log-odds, so the decision is at 0, not at 0.5
                    row = label_attack(model, DATASETS[dataset], reasons, x_te, seed, threshold=0.0)
                    save(rows, {**base, **key, **row, "attack": "label", "threshold": 0.0})


if __name__ == "__main__":
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    for dataset in sys.argv[1:] or list(DATASETS):
        run(dataset, rows)
