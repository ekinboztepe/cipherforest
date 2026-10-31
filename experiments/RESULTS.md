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
An input the copy got wrong becomes an extra starting point of the next round.
If a round finds nothing new, the client also runs the 2-d test from that
input on every output: an edge between f and g can show up only jointly, in a
third feature's SHAP value, which 1-d sweeps rarely see. 8-bit outputs almost
never need this; the coarse outputs below do.

| attacker | Adult queries, median (max) | Taiwan queries, median (max) |
|---|---|---|
| prediction only, every cell | 2547 (3123) | 28767 (30813) |
| prediction only, graph | 1382 (1640) | 2492 (5038) |
| prediction + SHAP, graph | 687 (1180) | 2148 (3198) |

Every copy matched the server on all 5000 inputs.

- A few thousand queries copy the model exactly even without explanations. The
  leak comes from the small model and the public edges, not from SHAP.
- Against an attacker that also exploits structure, SHAP saves 1.2x to 2x. The
  naive comparison (every cell) would suggest 4x to 13x, which overstates it.
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
| Adult | 0.5 | 0.896 | 0.996 | 0.95 | 256 | 9.4M |
| Adult | 1 | 0.894 | 0.987 | 0.89 | 512 | 30M |
| Adult | 2 | 0.889 | 0.953 | 0.81 | 2048 | 103M |
| Adult | 4 | 0.871 | 0.850 | 0.71 | 8192 | 609M |
| Taiwan | 0 | 0.763 (0.763) | 1.000 | 1.00 | 1 | 1621 |
| Taiwan | 0.5 | 0.762 | 0.994 | 0.89 | 256–512 | 16M |
| Taiwan | 1 | 0.760 | 0.977 | 0.83 | 1024 | 119M |
| Taiwan | 2 | 0.753 | 0.920 | 0.71 | 4096 | 274M |
| Taiwan | 4 | 0.738 | 0.773 | 0.57 | 16384 | 1.5G |

Copies matched the clean model on at least 99.5% of 5000 random inputs; a few
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
the ~700 to 2200 queries above, with noise we have not measured it. A plain
gradient-boosted regressor on the prediction, which needs no structure, is
the baseline. 2 seeds, budgets 2^10 to 2^20, scored on the test split.

Queries until every seed's copy agrees with the clean model on 99% of test
labels (threshold 0.5) / on every test prediction and SHAP vector after
rounding to the output grid. The repeating attacker's exact-copy cost from the
previous section is in the last column.

| data | b | tables, pred+shap | tables, pred | gbdt, pred | repeating |
|---|---|---|---|---|---|
| Adult | 0 | 1K / 4K | 1K / 4K | >1M / >1M | 687 |
| Adult | 0.5 | 1K / 16K | 1K / 66K | >1M / >1M | 9.4M |
| Adult | 1 | 4K / 66K | 4K / 262K | >1M / >1M | 30M |
| Adult | 2 | 4K / 262K | 16K / 1M | >1M / >1M | 103M |
| Taiwan | 0 | 1K / 1K | 1K / 1K | 4K / >1M | 1621 |
| Taiwan | 0.5 | 1K / 66K | 1K / 262K | 4K / >1M | 16M |
| Taiwan | 1 | 1K / 262K | 1K / 1M | 4K / >1M | 119M |
| Taiwan | 2 | 1K / 1M | 1K / >1M | 16K / >1M | 274M |

- Output noise does not protect the model. At b = 2, where the honest client
  already loses 0.01 AUC and a fifth of its top-3 features, an exact copy costs
  262K queries on Adult and 1M on Taiwan, 260 to 390 times fewer than repeating.
  A copy that gives the same decision 99% of the time costs a few thousand.
- SHAP still helps by about 4x here: each SHAP output carries one noise term,
  the prediction carries the sum of all of them.
- The generic regressor is much worse than the structured fit (1.5 to 1.9
  output units of error at 1M queries). Knowing the model's form is what makes
  the attack cheap, and client.json gives most of it away.
- Open: the cost of finding thresholds and the graph through noise. A
  threshold is a jump in the mean output along one feature, so the same
  averaging should find it, but this is not measured.

## Hiding or replacing the edges (`edges_bench.py`)

Three deployments of the same pipeline:

- quantile edges, public: the e2e model.
- uniform edges, public: edges evenly spaced between the training minimum and
  maximum, so they say nothing about the distribution in between.
- quantile edges, hidden: the server quantises raw inputs, and the client
  only knows each feature's range.

With hidden edges, the client sweeps each feature over 16 points and bisects
every interval whose ends answer differently, down to adjacent floats. Since
code = number of edges < x, the lower float is then the edge itself. It then
runs the attack above on the edges it found. 5 seeds.

| data | edges | AUC | pred+shap queries, median (max) | pred queries, median (max) |
|---|---|---|---|---|
| Adult | quantile, public | 0.897 | 687 (1180) | 1390 (1487) |
| Adult | uniform, public | 0.899 | 995 (2071) | 2290 (3397) |
| Adult | quantile, hidden | 0.897 | 2652 (2803) | 5265 (6385) |
| Taiwan | quantile, public | 0.763 | 2147 (2148) | 2497 (5038) |
| Taiwan | uniform, public | 0.755 | 3077 (3079) | 1522 (3434) |
| Taiwan | quantile, hidden | 0.763 | 8512 (8743) | 8930 (10427) |

Every copy matched the server on the whole test split and on 5000 random
inputs from the range, except two: uniform edges, Adult, prediction only, where
the attacker stopped (at 985 and 2168 queries) with a copy right on only 94% of
the test split and 99.9% of the range. Its own check samples codes evenly, and with even edges the real data
sit in a few of them. More checks would fix it; the median is unaffected.

- Hiding the edges multiplies the cost by about 4. Every run recovered all the
  edges the model uses (14 on Adult, 16 on Taiwan) bit for bit, and no wrong
  ones. So the hidden deployment still leaks those training quantiles; only
  the unused ones stay private. In TFHE it also moves quantisation onto the
  server, which is not free.
- Uniform edges keep the training distribution out of client.json at little
  AUC cost (+0.002 on Adult, -0.008 on Taiwan) but do not protect the model.
  They stop a data leak, not a model leak.
- Together with the noise results: for a model this small, the cheap defences
  only raise the price from hundreds to thousands of queries. What is left is
  a per-client query limit, or a bigger model, which the attack's 2^m
  inclusion-exclusion and the regression's table sizes would feel first.

## Coarser outputs (`coarse_bench.py`)

Fewer output bits per SHAP value: the same student quantised to 8, 6, 5 or 4
bits (3 trees need at least 4, since each can add one rounding unit). The
attacker copies the deployed coarse model, so the honest cost is measured
against the float student the owner trained. 5 seeds per attacker.

| data | bits | AUC | SHAP cosine vs float | labels vs float | pred+shap queries, median | pred queries, median |
|---|---|---|---|---|---|---|
| Adult | 8 | 0.897 | 0.999 | 1.000 | 687 | 1390 |
| Adult | 6 | 0.896 | 0.987 | 1.000 | 1763 | 1969 |
| Adult | 5 | 0.890 | 0.923 | 0.963 | 1805 | 2086 |
| Adult | 4 | 0.876 | 0.842 | 0.988 | 2898 | 3079 |
| Taiwan | 8 | 0.763 | 0.999 | 1.000 | 2147 | 2497 |
| Taiwan | 6 | 0.763 | 0.972 | 0.999 | 3315 | 2600 |
| Taiwan | 5 | 0.759 | 0.927 | 0.997 | 5332 | 3674 |
| Taiwan | 4 | 0.752 | 0.797 | 0.998 | 9615 | 5117 |

Every copy was exact on the test split except three at 4 bits on Taiwan
(99.6%, 95.9%, 99.6%): there the copy and the server rarely differ, and 200
check queries do not see it.

- Coarsening is the worst trade-off measured. At 4 bits the honest client
  loses 0.02 AUC and the explanations drift to cosine 0.8, while the attacker
  pays at most 4.5x more and still ends with an exact or near-exact copy.
- The extra cost comes from effects that only show jointly (see Extraction),
  not from hidden information: the coarse model is still a sum of clique
  tables, and nothing it computes is out of the attacker's reach.
- 6 bits is nearly free for the honest client and costs the attacker 1.5x to
  2.5x more. Not a defence on its own.
- Not measured: a label-only server, which the attack above does not cover.
