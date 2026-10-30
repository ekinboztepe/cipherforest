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

## Extraction, step 1: thresholds (`extraction_bench.py`)

A client with only `client.json` sweeps each feature through all its input codes
from a few random points and records where the output changes. This can only
hit true thresholds; the question is how many queries finding all of them takes.
Same e2e models, 5 seeds.

| | Adult (14 thresholds) | Taiwan (16 thresholds) |
|---|---|---|
| queries per starting point | 153 | 432 |
| recall, prediction + SHAP, 1 point | 1.00 every seed | 1.00 every seed |
| recall, prediction only, 1 point | 0.86 to 1.00 | 0.75 to 1.00 |
| recall, prediction only, 2 points | 1.00 every seed | 1.00 every seed |

Every threshold of the deployed model leaks in a few hundred queries, with or
without explanations. SHAP only halves the cost here, because in an oblivious
tree every split changes the leaf of every input. Recovering the leaf values
is the next step.
