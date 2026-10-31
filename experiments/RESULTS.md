# Results

## SHAP tables are exact

`tests/test_shap_table.py` compares the per-leaf table with brute-force
permutation Shapley values on 150 random trees (depth 1 to 5, repeated features,
random background sets). Equal to 1e-9, efficiency and additivity included.

## Distillation (`distill_bench.py`)

Teacher: random forest, 100 trees, max_depth 12. Mean over seeds 0 to 2,
25 % held out. Raw rows in `results/distill_bench.json`.

| dataset | teacher | obl_distill D6 T3 | cb_labels D6 T3 | dt_labels D6 | SHAP cosine vs teacher | top-3 overlap |
|---|---|---|---|---|---|---|
| adult | 0.918 | 0.898 | 0.903 | 0.890 | 0.81 | 0.68 |
| bank_marketing | 0.924 | 0.896 | 0.894 | 0.858 | 0.83 | 0.62 |
| breast_cancer | 0.983 | 0.975 | 0.975 | 0.923 | 0.58 | 0.40 |
| credit_default | 0.779 | 0.768 | 0.761 | 0.751 | 0.84 | 0.71 |
| credit_g | 0.786 | 0.723 | 0.736 | 0.701 | 0.82 | 0.66 |

AUC. Takeaways:

- Three depth-6 oblivious trees lose 0.01 to 0.03 AUC; credit_g (1000 noisy rows) loses 0.06.
- Distilling does not beat a same-size CatBoost trained on the labels. The distiller is not the contribution.
- Training on per-tree vote fractions (`obl_votes`) is no better than on probabilities.
- Student SHAP agrees only moderately with the teacher's. Worst on breast_cancer,
  where the teacher spreads credit over 30 correlated features and the student uses at most 13.

## End-to-end (`e2e_prepare.py` + `e2e_fhe.py`)

D=6, T=3, 32 quantiser bins, 8-bit signed outputs. Client and server exchange only bytes.

| | Adult | Taiwan credit default |
|---|---|---|
| features (used) | 14 (7) | 23 (13) |
| AUC teacher / student / quantised student | 0.918 / 0.896 / 0.897 | 0.774 / 0.763 / 0.763 |
| integer routing == float routing | 12 211 / 12 211 rows | 7 500 / 7 500 rows |
| encrypted == integer reference | 20 / 20 | 20 / 20 |
| PBS per request | 31 | 36 |
| server run, median (max) | 0.69 s (2.06 s) | 0.61 s (1.36 s) |
| encrypt / decrypt | 6.7 ms / 1.9 ms | 7.0 ms / 2.0 ms |
| keygen | 3.8 s | 2.4 s |
| evaluation keys | 566 MB | 335 MB |
| ciphertext in / out | 224 KB / 224 KB | 368 KB / 416 KB |
| max error vs float student, prediction / SHAP | 0.005 / 0.009 | 0.004 / 0.004 |

The max is the first request; medians are stable. Laptop, Concrete 2.11, Docker
Desktop on a memory-constrained WSL2 VM.

For scale only (different model and scheme, vendor numbers): CipherExplain lists
about 70 s per encrypted explanation for a 100-tree, depth-4 XGBoost under CKKS.

## Extraction (`extraction_bench.py`)

A client with only `client.json` and a query budget copies the deployed model
exactly: same prediction and SHAP vector on every input. Same e2e models,
10 seeds; the copy is then compared with the server on 5000 random inputs.

1. Thresholds. The client knows the edges, so it sweeps one feature through all
   its codes and sees where the output changes. That only happens at true
   thresholds.
2. Interaction graph: which features share a tree. With SHAP, phi_f moves when
   g moves. Without, a 2-d sweep of (f, g) is not additive.
3. Cells. The model is a sum over the maximal cliques of that graph, so a grid
   over each clique plus inclusion-exclusion gives it exactly. Without the
   graph, the client queries every cell.

Each round adds one random starting point; the client stops once its copy
agrees with the server on 200 fresh queries, which count towards the total.

| attacker | Adult queries, median (max) | Taiwan queries, median (max) |
|---|---|---|
| prediction only, every cell | 2624 (3088) | 28563 (31014) |
| prediction only, graph | 1169 (1670) | 2684 (6483) |
| prediction + SHAP, graph | 687 (1027) | 1722 (2348) |

Every copy matched the server on all 5000 inputs.

- A few thousand queries copy the model exactly even without explanations. The
  leak comes from the small model and the public edges, not from SHAP.
- Against an attacker that also exploits structure, SHAP saves about 1.6x. The
  naive comparison (every cell) would suggest 4x to 17x, which overstates it.
- Defences that only touch the SHAP output cannot stop this; the prediction
  alone suffices. Hiding the edges and coarsening the output are the candidates.

## Noise on the outputs (`defense_bench.py`)

The server adds rounded Laplace(b) noise, in output units, to every integer
SHAP output, and so to the prediction. In the circuit this is a clear integer
added to a ciphertext, no bootstrapping. The honest client asks once. The
attacker asks every input r times and rounds the means back onto the grid,
doubling r until its copy passes the 200-query check; all attempts count.
2 seeds; honest-client numbers on the e2e test split.

| data | b | AUC (clean) | SHAP cosine | top-3 overlap | repeats | attacker queries, median |
|---|---|---|---|---|---|---|
| Adult | 0 | 0.897 (0.897) | 1.000 | 1.00 | 1 | 687 |
| Adult | 0.5 | 0.896 | 0.996 | 0.95 | 128–256 | 8.5M |
| Adult | 1 | 0.894 | 0.987 | 0.89 | 512 | 21M |
| Adult | 2 | 0.889 | 0.953 | 0.81 | 2048 | 115M |
| Adult | 4 | 0.871 | 0.850 | 0.71 | 8192 | 382M |
| Taiwan | 0 | 0.763 (0.763) | 1.000 | 1.00 | 1 | 1408 |
| Taiwan | 0.5 | 0.762 | 0.994 | 0.89 | 256 | 24M |
| Taiwan | 1 | 0.760 | 0.977 | 0.83 | 1024 | 121M |
| Taiwan | 2 | 0.753 | 0.920 | 0.71 | 4096 | 452M |
| Taiwan | 4 | 0.738 | 0.773 | 0.57 | 16384 | 1.9G |

Copies matched the clean model on at least 99.6% of 5000 random inputs; a few
cells were rounded the wrong way and the 200-query check missed them.

- b = 0.5 costs the honest client almost nothing (AUC -0.001, cosine 0.99) and
  this attacker 10^4 times more queries. r grows like b^2, as expected.
- The server cannot spot repeats: every query is a fresh ciphertext, and it
  cannot key the noise to an input it never sees. So averaging is always
  possible and the noise only works together with a per-client query limit.
- Not differential privacy: there is no budget across queries.
- This attacker is naive: it repeats every input equally. The next section
  shows how much cheaper learning from the noise is.

## Learning through the noise (`learning_bench.py`)

Instead of repeating inputs, the attacker asks n random inputs once and fits a
copy. The model is a sum of one table per clique, so least squares on cell
indicators fits it, and every query informs every table. These attackers get
the thresholds and the interaction graph for free; without noise those cost
the ~700 to 1700 queries above, with noise we have not measured it. A plain
gradient-boosted regressor on the prediction, which needs no structure, is
the baseline. 2 seeds, budgets 2^10 to 2^20, scored on the test split.

Queries until every seed's copy agrees with the clean model on 99% of test
labels (threshold 0.5) / on every test prediction and SHAP vector after
rounding to the output grid. The repeating attacker's exact-copy cost from the
previous section is in the last column.

| data | b | tables, pred+shap | tables, pred | gbdt, pred | repeating |
|---|---|---|---|---|---|
| Adult | 0 | 1K / 4K | 1K / 4K | >1M / >1M | 687 |
| Adult | 0.5 | 1K / 16K | 1K / 66K | >1M / >1M | 8.5M |
| Adult | 1 | 4K / 66K | 4K / 262K | >1M / >1M | 21M |
| Adult | 2 | 4K / 262K | 16K / 1M | >1M / >1M | 115M |
| Taiwan | 0 | 1K / 1K | 1K / 1K | 4K / >1M | 1408 |
| Taiwan | 0.5 | 1K / 66K | 1K / 262K | 4K / >1M | 24M |
| Taiwan | 1 | 1K / 262K | 1K / 1M | 4K / >1M | 121M |
| Taiwan | 2 | 1K / 1M | 1K / >1M | 16K / >1M | 452M |

- Output noise does not protect the model. At b = 2, where the honest client
  already loses 0.01 AUC and a fifth of its top-3 features, an exact copy costs
  262K queries on Adult and 1M on Taiwan, about 440 times fewer than repeating.
  A copy that gives the same decision 99% of the time costs a few thousand.
- SHAP still helps by about 4x here: each SHAP output carries one noise term,
  the prediction carries the sum of all of them.
- The generic regressor is much worse than the structured fit (1.5 to 1.9
  output units of error at 1M queries). Knowing the model's form is what makes
  the attack cheap, and client.json gives most of it away.
- Open: the cost of finding thresholds and the graph through noise. A
  threshold is a jump in the mean output along one feature, so the same
  averaging should find it, but this is not measured.
