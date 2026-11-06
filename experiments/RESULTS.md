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

The max is the first request; medians are stable. Desktop (Ryzen 5 3600, 16 GB), Concrete 2.11, Docker
Desktop on a memory-constrained WSL2 VM.

Bank marketing came later and only went through the defence benches below: 16
features (8 used), AUC teacher / student / quantised student 0.925 / 0.899 /
0.899, max error 0.006 on both outputs. Its encrypted runs are in the FHE section.

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

| attacker | Adult queries, median (max) | Taiwan queries, median (max) | Bank marketing queries, median (max) |
|---|---|---|---|
| prediction only, every cell | 2547 (3123) | 28767 (30813) | 2962 (3885) |
| prediction only, graph | 1382 (1640) | 2492 (5038) | 963 (1717) |
| prediction + SHAP, graph | 687 (1180) | 2148 (3198) | 864 (1442) |

Every copy matched the server on all 5000 inputs, except four of the ten
prediction + SHAP copies on Bank marketing (98.7% to 99.7%). Those stopped at 717
to 800 queries, when the 200-query check happened not to hit the few cells they
got wrong; the six exact copies took 859 to 1442. So 864 is a little optimistic
for an exact copy, and a longer check would close the gap.

- A few thousand queries copy the model exactly even without explanations. The
  leak comes from the small model and the public edges, not from SHAP.
- Against an attacker that also exploits structure, SHAP saves 1.1x to 2x. The
  naive comparison (every cell) would suggest 3x to 13x, which overstates it.
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
| Bank marketing | 0 | 0.899 (0.899) | 1.000 | 1.00 | 1 | 763 |
| Bank marketing | 0.5 | 0.897 | 0.992 | 0.93 | 256 | 31M |
| Bank marketing | 1 | 0.894 | 0.971 | 0.87 | 512–1024 | 126M |
| Bank marketing | 2 | 0.884 | 0.904 | 0.77 | 4096 | 462M |
| Bank marketing | 4 | 0.865 | 0.755 | 0.65 | 8192 | 2.5G |

Copies matched the clean model on at least 99.5% of 5000 random inputs on Adult
and Taiwan, 98.7% on Bank marketing; the 200-query check missed the rest, as in
the section above.

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
| Bank marketing | 0 | 1K / 4K | 1K / 4K | 66K / >1M | 763 |
| Bank marketing | 0.5 | 1K / 66K | 1K / 262K | 16K / >1M | 31M |
| Bank marketing | 1 | 4K / 66K | 1K / 262K | 66K / >1M | 126M |
| Bank marketing | 2 | 4K / 262K | 4K / >1M | 66K / >1M | 462M |

- Output noise does not protect the model. At b = 2, where the honest client
  already loses 0.01 AUC and a fifth of its top-3 features, an exact copy costs
  262K queries on Adult and Bank marketing and 1M on Taiwan, 260 to 1800 times
  fewer than repeating.
  A copy that gives the same decision 99% of the time costs a few thousand.
- SHAP still helps by about 4x here: each SHAP output carries one noise term,
  the prediction carries the sum of all of them.
- The generic regressor is much worse than the structured fit (1.5 to 2.2
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
runs the attack above on the edges it found. If the copy fails a check on random
inputs from the range, it bisects again from an input that failed. 5 seeds.

| data | edges | AUC | pred+shap queries, median (max) | pred queries, median (max) |
|---|---|---|---|---|
| Adult | quantile, public | 0.897 | 687 (1180) | 1390 (1487) |
| Adult | uniform, public | 0.899 | 995 (2071) | 2290 (3397) |
| Adult | quantile, hidden | 0.897 | 2652 (2803) | 4263 (4360) |
| Taiwan | quantile, public | 0.763 | 2147 (2148) | 2497 (5038) |
| Taiwan | uniform, public | 0.755 | 3077 (3079) | 1522 (3434) |
| Taiwan | quantile, hidden | 0.763 | 8512 (8743) | 8930 (10467) |
| Bank marketing | quantile, public | 0.899 | 728 (1442) | 1591 (1673) |
| Bank marketing | uniform, public | 0.895 | 1224 (2438) | 2518 (5426) |
| Bank marketing | quantile, hidden | 0.899 | 4973 (5166) | 6351 (6950) |

Every copy matched the server on the whole test split and on 5000 random
inputs from the range, except two groups. Uniform edges, Adult, prediction only:
two copies stopped (at 985 and 2168 queries) right on only 94% of the test split
and 99.9% of the range. Quantile edges, public, Bank marketing, with SHAP: four
copies right on 98.4% to 99.6% of the test split and all of the range. The
attacker's own check samples codes evenly, while real data sit in a few of them.
More checks would fix both; on Bank marketing the median is what moves.

A first version of the hidden attacker always bisected from a random start, and on
Bank marketing one prediction-only run gave up after 10 rounds, one threshold
short and aware of it. That was the attacker's weakness, not a defence, so it now
starts from an input its copy got wrong. The numbers above are with that fix.

- Hiding the edges multiplies the cost by about 3 to 7. Every run recovered all the
  edges the model uses (14 on Adult, 16 on Taiwan, 15 on Bank marketing) bit for bit, and no wrong
  ones. So the hidden deployment still leaks those training quantiles; only
  the unused ones stay private. In TFHE it also moves quantisation onto the
  server, which is not free.
- Uniform edges keep the training distribution out of client.json at little
  AUC cost (+0.002 on Adult, -0.008 on Taiwan, -0.004 on Bank marketing) but do not protect the model.
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
| Bank marketing | 8 | 0.899 | 0.997 | 1.000 | 728 | 1591 |
| Bank marketing | 6 | 0.897 | 0.956 | 0.997 | 1984 | 1922 |
| Bank marketing | 5 | 0.895 | 0.880 | 0.999 | 2236 | 2614 |
| Bank marketing | 4 | 0.871 | 0.879 | 0.991 | 3890 | 3631 |

Every copy was exact on the test split except three at 4 bits on Taiwan
(99.6%, 95.9%, 99.6%) and eight on Bank marketing (four at 8 bits, one at 6, one
at 5, two at 4; 98.4% or better): there the copy and the server rarely differ,
and 200 check queries do not see it.

- Coarsening is the worst trade-off measured. At 4 bits the honest client
  loses 0.01 to 0.03 AUC and the explanations drift to cosine 0.8 to 0.88, while the attacker
  pays at most about 5x more and still ends with an exact or near-exact copy.
- The extra cost comes from effects that only show jointly (see Extraction),
  not from hidden information: the coarse model is still a sum of clique
  tables, and nothing it computes is out of the attacker's reach.
- 6 bits is nearly free for the honest client and costs the SHAP attacker 1.5x
  to 2.7x more. Not a defence on its own.
- Not measured: a label-only server, which the attack above does not cover.

## Returning only the top-k SHAP values

Not implemented, because the benches above already bound it. As long as the
prediction comes back, the attacker can ignore the explanation altogether, and
that attacker is the prediction-only one: about 1000 to 2500 queries for a copy
of the prediction on public edges. So top-k can at best take the SHAP attacker
up to that cost, a factor of 1.1 to 2 by the extraction table. Once it holds the
model, the attacker can also compute SHAP values itself, against a background set
of its own. The honest client, on the other hand, loses the other values and the
check that they sum to the prediction. A server that also
withholds the prediction is the label-only case above, not measured.

## What the defences cost under FHE (`defense_fhe_prepare.py` + `defense_fhe.py`)

The e2e circuit with each defence built in, compiled with Concrete 2.11 in the
same Docker setup. Noise and coarse outputs ran encrypted on 10 samples, every
result equal to the integer reference plus the noise the server drew. Hidden
edges are compile-time estimates only (see below). Cost is Concrete's
complexity estimate relative to the e2e circuit.

- noise-b: the server passes rounded Laplace(b), clipped to +-8b, as a clear
  argument and the circuit adds it to the outputs.
- bits-k: the student quantised to k output bits.
- grid-p: hidden edges. The client encodes each feature on a public p-bit
  grid over the dataset's range, and every threshold moves up to the next grid
  edge. Up, not to the nearest: discrete features have many rows exactly on an
  edge, and rounding down flipped 40% of Taiwan's labels. Labels are compared
  with the deployed model on the test split.

| data | variant | PBS | widest value | cost | evaluation keys | ciphertext in | server run, median | labels vs deployed |
|---|---|---|---|---|---|---|---|---|
| Adult | e2e | 31 | 8 bits | 1x | 566 MB | 224 KB | 0.74 s | 1 |
| Adult | noise b = 0.5 / 2 / 4 | 31 | 9 | | 735 MB | 224 KB | 0.9 to 1.1 s | |
| Adult | bits-6, bits-4 | 31 | 6 | | 566 MB | 224 KB | 0.7 s | 1.000 / 0.988 |
| Adult | grid-8 | 31 | 8 | 7.6x | 2.2 GB* | 1.8 MB | | 0.998 |
| Adult | grid-10 | 31 | 10 | 37x | 11 GB* | 7 MB | | 0.999 |
| Adult | grid-12 | 27 | 12 | 136x | (0.8 GB*) | 0.9 MB | | 1 |
| Adult | grid-16 | 27 | 16 | 200x | (0.9 GB*) | 1.1 MB | | 1 |
| Taiwan | e2e | 36 | 8 | 1x | 335 MB | 368 KB | 0.68 s | 1 |
| Taiwan | noise b = 0.5 / 2 / 4 | 36 | 9 | 1.3x | 468 MB | 368 KB | 0.9 to 1.2 s | |
| Taiwan | bits-6, bits-4 | 36 | 6 | 1x | 335 MB | 368 KB | 0.7 to 0.8 s | 0.999 / 0.998 |
| Taiwan | grid-8 | 36 | 8 | 6x | 2.2 GB* | 2.9 MB | | 0.983 |
| Taiwan | grid-10 | 36 | 10 | 29x | 11 GB* | 12 MB | | 0.994 |
| Taiwan | grid-12 | 34 | 12 | 129x | (0.8 GB*) | 1.5 MB | | 0.997 |
| Taiwan | grid-16 | 34 | 16 | 189x | (0.9 GB*) | 1.8 MB | | 1 |
| Bank marketing | e2e | 32 | 8 | 1x | 335 MB | 256 KB | 1.9 s† | 1 |
| Bank marketing | noise b = 0.5 / 2 / 4 | 32 | 9 | 1.3x | 468 MB | 256 KB | 2.1 s† | |
| Bank marketing | bits-6, bits-4 | 32 | 6 | 1x | 335 MB | 256 KB | 1.9 s† | 0.997 / 0.991 |
| Bank marketing | grid-8 | 32 | 8 | 6.7x | 2.2 GB* | 2 MB | | 0.997 |
| Bank marketing | grid-10 | 32 | 10 | 32x | 11 GB* | 8 MB | | 1.000 |
| Bank marketing | grid-12 | 29 | 12 | 128x | (0.8 GB*) | 1 MB | | 1 |
| Bank marketing | grid-16 | 29 | 16 | 187x | (0.9 GB*) | 1.3 MB | | 1 |

† Capped at 2 CPUs, unlike the Adult and Taiwan rows, so compare within the
dataset only. Bank marketing's grids were compiled only; we did not try keygen.

\* Concrete's estimate of bootstrap plus keyswitch keys, not measured. For
12 bits and up Concrete switches to another bootstrapping method with smaller
keys of those two kinds, but keygen still went past 3 GB on both runs we
tried (Adult grid-12, Taiwan grid-16), so the real total is larger. Concrete
also could not lower the vectorised comparison on inputs wider than 11 bits;
those circuits compare one feature at a time, which saves a few PBS.

- Noise is cheap: one more output bit, 30% to 40% larger keys and up to 0.3 s
  more per request, whatever the bound. It is also the defence the learning
  attacker removes (see above).
- Coarser outputs cost the server nothing and save it nothing: the input
  comparisons, not the output tables, set the parameters.
- Hidden edges are the only defence that costs FHE much, and their price grows
  with the grid. Adult keeps every prediction only at 12 bits, Bank marketing every
  label at 12, Taiwan every label only at 16, about 130x, 130x and 190x the e2e
  circuit's cost. The Adult and Taiwan keys did not fit this machine. In exchange
  the attacker pays about 3x to 7x more queries.
- Taken together: none of the defences buys much against extraction, and the
  one that buys the most is by far the most expensive under FHE. For a model
  this small, a per-client query limit is the defence that remains.

## Against Concrete-ML (`concrete_ml_bench.py`)

Concrete-ML 1.9's own XGBClassifier (depth 6, 6-bit inputs, prediction only)
against our pipeline (T oblivious trees of depth 6, 5-bit inputs, 8-bit outputs,
prediction and exact SHAP). Same split, same Concrete 2.10 from
`docker/concrete-ml.Dockerfile`, each through its own client/server files. Our
circuit was compiled twice: with the settings Concrete-ML uses (compressed keys
and inputs, p_error 2^-40 per bootstrap), and with Concrete's defaults as in the
e2e demo.

Every container was capped at 2 CPUs and 3 GB, so times here are slower than in
the e2e table. Server run is the median over 10 rows, 3 for XGB T=30. Our
results equalled the integer reference on every row, Concrete-ML's its own
simulation. T=100 was compiled only; the last column is the ratio of Concrete's
complexity estimates, XGB over ours on the same settings.

| data | T | PBS XGB / ours | AUC XGB / ours | server run XGB | ours, same settings | ours, defaults | cost ratio |
|---|---|---|---|---|---|---|---|
| Adult | 3 | 1218 / 31 | 0.902 / 0.897 | 19.5 s | 3.6 s | 2.1 s | 5.5x |
| Adult | 10 | 4690 / 109 | 0.912 / 0.906 | 60.6 s | 9.9 s | 5.3 s | 5.3x |
| Adult | 30 | 15960 / 338 | 0.919 / 0.904 | 183 s | 34.5 s | 16.2 s | 5.6x |
| Adult | 100 | 56800 / 1137 | 0.919 / 0.895 | | | | 5.8x |
| Taiwan | 3 | 1461 / 36 | 0.761 / 0.763 | 25.6 s | 4.3 s | 2.0 s | 4.4x |
| Taiwan | 10 | 5050 / 117 | 0.767 / 0.769 | 73.1 s | 11.7 s | 5.8 s | 4.7x |
| Taiwan | 30 | 15150 / 353 | 0.766 / 0.766 | 149 s* | 33.6 s | 16.5 s | 4.7x |
| Taiwan | 100 | 51400 / 1182 | 0.759 / 0.752 | | | | 4.7x |

\* Third attempt. The first produced no output in 30 minutes, the second
finished one row in 186 s and not the next in the rest of its 30 minutes. An
earlier Adult XGB T=10 run also stalled for an hour, then took 10 minutes on the
retry. We do not know why, so treat the XGB times as numbers from this one machine.

|  | evaluation keys | ciphertext in |
|---|---|---|
| Concrete-ML XGB | 74 MB | 0.33 KB (Adult), 0.54 KB (Taiwan) |
| ours, same settings | 361 MB (Adult), 319 MB (Taiwan) | same |
| ours, defaults | 566 to 829 MB (Adult), 335 MB (Taiwan) | 224 KB (Adult), 368 KB (Taiwan) |

- Ours is about 4x to 6x faster on these two datasets on the same settings, and
  returns the explanation too. The server runs below give a wider range. The PBS count is 40x to 50x lower, but that overstates it: our
  bootstraps carry heavier parameters, and time follows Concrete's complexity
  estimate, not the PBS count.
- Concrete-ML's keys are 4x to 5x smaller than ours on the same settings.
- Compression is free to take: it shrinks our ciphertext about 700x, for about
  twice the server time because of the stricter p_error.
- Ours does not gain from more trees. Adult's float student stays at 0.906 from
  T=10 on while XGB climbs to 0.919, so at T=30 XGB is ahead by 0.015 and buys it
  with 5x the time. On Taiwan the two are level.
- At T=100 our 8 output bits start to hurt: the float student scores 0.907 on
  Adult and 0.767 on Taiwan, the quantised one 0.895 and 0.752.
- Adult ours T=10 with defaults picked larger keys than T=3 or T=30 and was
  killed at 3 GB; its row was run with 4 GB. Compressed keys avoid this: the same
  model on Concrete-ML's settings fit in 3 GB.

## On a rented server (`server_bench.py`)

Three more datasets and five seeds. The seed changes the split, the requests and
the SHAP background. Raw rows are in `results/server_bench.csv` (16 vCPU) and
`results/server_bench_8vcpu.csv` (8 vCPU).

| machine | CPU | used for |
|---|---|---|
| Runpod cpu3c, 16 vCPU, 32 GB, EUR-IS-1 | EPYC 9965 | ours with Concrete's defaults |
| Runpod cpu3c, 8 vCPU, 16 GB, EU-RO-1 | EPYC 9655P | ours on Concrete-ML's settings against Concrete-ML |

These are cloud containers on hosts shared with others, so treat the times as
numbers from those machines. Concrete-ML's own XGB T=30 on Adult took 18.7 s and
24.4 s on the same machine for the same circuit size.

### Against Concrete-ML, same machine, same settings

100 requests per job for ours, 10 for XGB; XGB's P99 is not reported because 10
requests cannot give one. Server run is the median of the per-job medians, then
the lowest and highest job. P99 leaves out the first request, which is slower and
has its own column. XGB keys are 74 MB everywhere. Five seeds, except XGB at T=30
with three (see below).

| data | T | AUC XGB / ours | server run XGB, s | ours, s | XGB / ours | P99 ours, s | first request ours, s | keys ours, MB |
|---|---|---|---|---|---|---|---|---|
| Adult | 3 | 0.903 / 0.899 | 2.58 (2.52-2.68) | 0.71 (0.67-0.73) | 3.6x | 0.79 | 2.51 | 361 |
| Taiwan | 3 | 0.763 / 0.767 | 3.14 (3.10-3.75) | 0.77 (0.76-0.79) | 4.1x | 0.98 | 2.32 | 319 |
| Bank | 3 | 0.896 / 0.897 | 3.47 (3.36-3.78) | 0.76 (0.56-0.78) | 4.6x | 0.83 | 2.14 | 215-319 |
| Breast cancer | 3 | 0.984 / 0.973 | 1.24 (1.02-1.29) | 0.77 (0.75-0.79) | 1.6x | 0.85 | 2.10 | 319 |
| German credit | 3 | 0.756 / 0.729 | 2.26 (1.92-2.59) | 0.74 (0.56-0.77) | 3.0x | 0.76 | 2.23 | 215-361 |
| Adult | 10 | 0.913 / 0.907 | 7.62 (7.15-7.95) | 1.84 (1.75-1.88) | 4.1x | 2.07 | 3.68 | 361 |
| Taiwan | 10 | 0.770 / 0.772 | 8.33 (8.03-8.46) | 1.95 (1.93-1.98) | 4.3x | 2.24 | 3.25 | 319 |
| Bank | 10 | 0.916 / 0.912 | 9.09 (8.61-9.24) | 1.94 (1.85-1.98) | 4.7x | 2.21 | 3.49 | 319 |
| Breast cancer | 10 | 0.988 / 0.972 | 2.18 (2.11-2.27) | 1.94 (1.91-2.02) | 1.1x | 2.25 | 3.59 | 319 |
| German credit | 10 | 0.779 / 0.732 | 5.27 (5.08-5.80) | 1.89 (1.84-2.05) | 2.8x | 2.22 | 3.81 | 347-361 |
| Adult | 30 | 0.920 / 0.907 | 18.67 (18.52-24.42) | 5.29 (5.13-5.68) | 3.5x | 6.87 | 7.46 | 361 |
| Taiwan | 30 | 0.771 / 0.766 | 18.97 (17.71-22.69) | 5.52 (5.33-5.95) | 3.4x | 6.51 | 8.71 | 319 |
| Bank | 30 | 0.929 / 0.913 | 23.22 (20.40-25.58) | 5.66 (5.18-5.90) | 4.1x | 6.34 | 7.88 | 319 |
| Breast cancer | 30 | 0.986 / 0.973 | 4.05 (3.61-4.14) | 5.49 (5.22-5.65) | 0.7x | 6.29 | 7.09 | 319 |
| German credit | 30 | 0.772 / 0.715 | 12.48 (12.06-13.60) | 5.33 (4.97-5.95) | 2.3x | 6.55 | 8.13 | 361-396 |

- On the three largest datasets ours is 3.4x to 4.7x faster at every T.
- The gap shrinks on the smaller ones: 2.3x to 3.0x on German credit, and on Breast
  cancer 1.6x, 1.1x, then 0.7x, so at T=30 Concrete-ML is the faster one. Its PBS
  count follows the size of its trees (2730 to 3000 there, against about 16000 on
  Adult) while ours stays near 340 whatever the data.
- At T=30 Concrete-ML has the higher AUC on all five datasets, by 0.005 to 0.057.
  From T=10 to T=30 ours gains at most 0.001 and on two datasets loses AUC.
- Concrete-ML returns only the prediction, ours the explanation too. Its keys are
  3x to 5x smaller.

XGB T=30 seeds 3 and 4 are missing. The first run's rows were lost: the boot script
stopped the pod when the queue ended, and that deleted the log, the only copy. We
then tried again on two other machines. EPYC 9654: Adult seed 3 gave no result in
19 minutes, and none in 11 after a restart. EPYC 9754: Adult and Taiwan seed 3 hit
the 20-minute limit, and Bank gave none in 11. The same jobs at seeds 0 to 2 took
4 to 6 minutes on the EPYC 9655P. We do not know whether these runs were slow or
stuck, as with the stalls in the 2 CPU runs above, and ours was not run on those
two machines. So the ratios hold for the EPYC 9655P only.

### Wrong results with Concrete's defaults

With the defaults ours is 2x to 3x faster and has larger, uncompressed keys:

| data | T | seeds | AUC teacher / float student / encrypted | server run, s | keys, MB |
|---|---|---|---|---|---|
| Adult | 3 | 5 | 0.918 / 0.899 / 0.899 | 0.35 (0.31-0.36) | 566-566 |
| Taiwan | 3 | 5 | 0.775 / 0.767 / 0.767 | 0.30 (0.27-0.32) | 335-335 |
| Bank | 3 | 5 | 0.925 / 0.897 / 0.897 | 0.31 (0.24-0.32) | 335-335 |
| Breast cancer | 3 | 5 | 0.984 / 0.974 / 0.973 | 0.30 (0.24-0.31) | 335-335 |
| German credit | 3 | 5 | 0.795 / 0.729 / 0.729 | 0.31 (0.24-0.35) | 335-566 |
| Adult | 10 | 5 | 0.918 / 0.907 / 0.907 | 0.75 (0.72-0.79) | 566-829 |
| Taiwan | 10 | 5 | 0.775 / 0.772 / 0.772 | 0.77 (0.59-0.79) | 335-463 |
| Bank | 10 | 5 | 0.925 / 0.913 / 0.912 | 0.76 (0.60-0.80) | 335-463 |
| Breast cancer | 10 | 5 | 0.984 / 0.973 / 0.972 | 0.69 (0.61-0.80) | 463-463 |
| German credit | 10 | 4 | 0.793 / 0.735 / 0.735 | 0.78 (0.76-0.83) | 532-566 |
| Adult | 30 | 3 | 0.918 / 0.908 / 0.906 | 1.93 (1.80-2.05) | 566-566 |
| Taiwan | 30 | 3 | 0.778 / 0.774 / 0.771 | 2.11 (2.11-2.15) | 335-335 |
| Bank | 30 | 3 | 0.924 / 0.919 / 0.912 | 2.03 (2.00-2.07) | 335-335 |
| Breast cancer | 30 | 3 | 0.984 / 0.974 / 0.973 | 2.46 (2.39-2.64) | 468-468 |
| German credit | 30 | 1 | 0.802 / 0.696 / 0.699 | 2.18 (2.18-2.18) | 747-747 |

Two jobs failed the check against the integer reference and gave no row: German
credit seed 1 T=10 at request 95, German credit seed 0 T=30 at request 9. T=30 stops
at seed 2 because we ended this run early to switch to Concrete-ML's settings, which
also cut German credit seed 2.

We replayed both rows on the desktop with fresh keys, twice each, and got the right
answer. All 250 test rows fit the chosen bit width, so it is not an overflow. The
likely cause, not proven, is Concrete's failure probability. The defaults allow
about 9.3e-6 per lookup, which the compiler reports as this per request:

| T | PBS | failure chance per request | requests | wrong |
|---|---|---|---|---|
| 3 | 31 | 2.9e-4 | 2500 | 0 |
| 10 | 115 | 1.0e-3 | 2996 | 1 |
| 30 | 338 to 353 | 3.2e-3 | 1310 | 1 |

On Concrete-ML's settings (p_error 2^-40 per bootstrap) ours had no mismatch in 7500
requests. Concrete 2.10's `circuit.global_p_error` returns the per-lookup value; the
real one is in `circuit.server._library.get_program_compilation_feedback()`.

So the fair comparison with Concrete-ML is its settings, not the defaults, and those
settings are the ones to deploy. Our circuit still compiles with the defaults unless
told otherwise.

More cores did not visibly help: ours at T=10 with defaults took 0.65 to 0.73 s on
8 vCPU and 0.69 to 0.78 s on 16, though the CPUs differ, so this settles nothing.

## Serve CatBoost directly (`catboost_bench.py`)

The student's accuracy loss came from its size, not from distilling (same-size
CatBoost on the labels scored the same, Distillation above). So: does a bigger
CatBoost model, served as it is, close the gap, and what does it cost? AUC of the
integer model the circuit serves, on the test split, mean over seeds 0 to 2. PBS is
the SHAP circuit's bootstrap count (depth + used features per tree).

| data | forest | distilled D6 T3 | distilled D6 T10 | distilled D6 T30 | CatBoost D4 T10 | CatBoost D4 T30 | CatBoost D4 T100 | CatBoost D6 T30 | CatBoost D6 T100 |
|---|---|---|---|---|---|---|---|---|---|
| Adult | 0.918 | 0.898 | 0.906 | 0.905 | 0.911 | 0.921 | 0.924 | 0.923 | 0.926 |
| Taiwan | 0.779 | 0.768 | 0.774 | 0.771 | 0.773 | 0.777 | 0.774 | 0.776 | 0.763 |
| Bank | 0.924 | 0.896 | 0.911 | 0.910 | 0.906 | 0.925 | 0.930 | 0.929 | 0.928 |
| Breast cancer | 0.983 | 0.974 | 0.974 | 0.971 | 0.983 | 0.990 | 0.989 | 0.988 | 0.989 |
| German credit | 0.786 | 0.723 | 0.739 | 0.725 | 0.770 | 0.785 | 0.793 | 0.777 | 0.778 |
| PBS (mean) | | 34 | 113 | 345 | 77 | 229 | 758 | 336 | 1114 |

- CatBoost at depth 4 with 30 trees matches the 100-tree forest on all five datasets
  (0.002 below to 0.007 above), with no distillation and its own exact SHAP values.
  The best distilled student of each dataset stays 0.005 to 0.047 below.
- It is also the cheaper circuit: about 230 bootstraps against 340 for 30 distilled
  trees of depth 6, since depth 4 needs fewer comparisons and each tree uses fewer
  features. Timed on Adult, seed 0, 20 requests each, in the same Docker image on
  the desktop (4 CPUs): 13.1 s for CatBoost depth 4 x 30 (AUC 0.921) against 17.0 s
  for distilled depth 6 x 30 (0.905); the score output takes 12.1 s.
- 100 trees buy little and start to lose to output rounding (8 bits): Taiwan drops
  from 0.776 to 0.763 at depth 6.
- So distil only what CatBoost cannot replace, a model you are handed rather than
  one you train; then use the teacher-fitted tables above for its explanations.

## Explanations fitted to the teacher (`align_bench.py`)

The student's exact SHAP values explain the student. When it stands in for a
teacher, users want the teacher's reasons, and the two agree only moderately.
`align_tables` keeps the splits and every leaf's total, so the prediction does not
move by one unit, and refits how each leaf's total is divided among the tree's
features to the teacher's SHAP values, by least squares on 1000 training rows with
a ridge towards the exact tables. The circuit is the same size: same players, same
leaves. Teacher: the 100-tree forest, interventional SHAP against the same 50
background rows as the student. Agreement on 200 test rows, mean over seeds 0 to 2,
ridge 1 (0.1 and 10 are in the json; 10 is a little worse, 0.1 the same).

| data | T | cosine, exact / aligned | top-3 overlap, exact / aligned | relative L1 error, exact / aligned |
|---|---|---|---|---|
| Adult | 3 | 0.80 / 0.84 | 0.67 / 0.70 | 0.75 / 0.55 |
| Adult | 10 | 0.84 / 0.95 | 0.69 / 0.87 | 0.64 / 0.25 |
| Taiwan | 3 | 0.86 / 0.89 | 0.73 / 0.77 | 0.65 / 0.51 |
| Taiwan | 10 | 0.89 / 0.94 | 0.73 / 0.81 | 0.56 / 0.39 |
| Bank | 3 | 0.83 / 0.90 | 0.62 / 0.74 | 0.83 / 0.48 |
| Bank | 10 | 0.86 / 0.95 | 0.67 / 0.80 | 0.68 / 0.32 |
| Breast cancer | 3 | 0.57 / 0.67 | 0.40 / 0.54 | 1.43 / 0.94 |
| Breast cancer | 10 | 0.57 / 0.82 | 0.41 / 0.70 | 1.45 / 0.62 |
| German credit | 3 | 0.81 / 0.84 | 0.66 / 0.69 | 0.84 / 0.67 |
| German credit | 10 | 0.78 / 0.83 | 0.64 / 0.67 | 1.03 / 0.73 |

- At T=10 the fitted tables agree with the teacher at cosine 0.82 to 0.95, against
  0.57 to 0.89 for the exact ones, and the L1 error falls by 30% to 60%. At T=3 the gain is
  smaller: three trees use few features, and a feature no tree uses gets 0 either way.
- What is given up: these are no longer the student's Shapley values. They are its
  prediction, split the way the teacher splits it, as far as the trees allow. They
  still add up to the prediction, and a feature no tree tests still gets 0.
- Breast cancer stays the hardest: the teacher spreads credit over 30 correlated
  features and the trees see at most 13 at T=3.

## What each output costs (`output_bench.py`)

`compile_model(output=...)` on the e2e student of each dataset (seed 0) with
`SAFE`, against Concrete-ML's XGBClassifier, compiled only. Server keys are the
expanded bootstrap and keyswitch keys the server holds, as Concrete reports them;
XGB's are 911 MB everywhere.

| data | T | PBS: XGB / shap / score / decision | server keys, MB: shap and score / decision | complexity, XGB over shap / over score |
|---|---|---|---|---|
| Adult | 3 | 1218 / 31 / 21 / 28 | 1174 / 2189 | 5.5 / 12.3 |
| Adult | 10 | 4690 / 109 / 70 / 77 | 1174 / 2189 | 5.3 / 11.8 |
| Adult | 30 | 15960 / 338 / 210 / 218 | 1174 / 2316 | 5.6 / 12.8 |
| Taiwan | 3 | 1461 / 36 / 21 / 28 | 981 / 1540 | 4.4 / 10.2 |
| Taiwan | 10 | 5050 / 117 / 70 / 78 | 981 / 1657 | 4.7 / 10.5 |
| Taiwan | 30 | 15150 / 353 / 210 / 218 | 981 / 1659 | 4.7 / 10.5 |
| Bank | 3 | 1623 / 32 / 21 / 28 | 981 / 1540 | 5.8 / 11.3 |
| Bank | 10 | 5410 / 107 / 70 / 77 | 981 / 1647 | 5.7 / 11.3 |
| Bank | 30 | 16230 / 337 / 210 / 217 | 981 / 1647 | 5.4 / 11.3 |
| Breast cancer | 3 | 246 / 34 / 21 / 28 | 981 / 1456 | 0.8 / 1.7 |
| Breast cancer | 10 | 820 / 114 / 70 / 77 | 981 / 1456 | 0.8 / 1.7 |
| Breast cancer | 30 | 2460 / 332 / 203 / 211 | 981 / 1552 | 0.8 / 1.8 |
| German credit | 3 | 948 / 36 / 21 / 28 | 1271 / 2117 | 3.2 / 8.4 |
| German credit | 10 | 3250 / 118 / 70 / 78 | 1297 / 2314 | 3.2 / 7.6 |
| German credit | 30 | 10560 / 352 / 210 / 218 | 1561 / 2687 | 3.5 / 8.2 |

Concrete's complexity estimate is not a stopwatch. Against the times measured in
server_bench it puts our lead over XGB at 5.3x to 5.6x on Adult where the server
measured 3.6x to 4.2x, and at 0.8x on Breast cancer where the server measured 1.35x,
1.09x and 0.69x for T = 3, 10, 30. So we timed the outputs themselves: 20 requests
each through the exported files on Adult, Docker on the desktop (4 CPUs, Ryzen 5
3600), medians without counting anything but `server.run`:

| T | shap | score | decision |
|---|---|---|---|
| 3 | 2.40 s | 2.01 s | 2.33 s |
| 30 | 17.11 s | 13.89 s | 13.96 s |

- The explanation is cheap. Dropping it saves 37% of the bootstraps at T=30 but only
  about 20% of the time (17.1 s against 13.9 s), and 16% at T=3. Exact SHAP costs
  about a quarter more than the prediction alone, not twice as much as the estimate
  suggests.
- At T=30 the decision costs the same time as the score, at T=3 its 7 extra
  bootstraps add 16%: the comparison runs on 4 bits after flooring the difference,
  which keeps its sign exact. Its keys are larger
  (1.5 to 2.7 GB expanded), because that last bootstrap needs its own parameters.
- Against Concrete-ML on the same output, the prediction alone, we did not time the
  two on one machine. From server_bench (3.4x to 4.7x with SHAP on the three large
  datasets) and the 1.2x above, expect roughly 4x to 6x there; on Breast cancer
  at T=30 Concrete-ML stays faster.

## Label-only answers (`label_bench.py`)

The defences above leave the prediction in the answer. A lender under ECOA
Regulation B need not: it can answer approve or decline, and on a decline name the
principal reasons. `decision_oracle` is that server on the e2e models: adverse is a
score at or below 0.5 on Adult and Bank marketing and above 0.5 on Taiwan, and with
reasons a decline names the 4 features that pushed hardest towards it, ranked, no
values. The "decision" output of `compile_model` computes the bit under encryption;
reasons would need an encrypted top-k and are not implemented in the circuit.

The attacker works in rounds of about 800 queries. It asks the 4 points its copy
is least sure about, sweeps 4 features through all their codes from each (a
decline's reasons if it has them, random used features otherwise), adds the 256
least certain of 8192 fresh random inputs, and refits a gradient-boosted copy on
every answer so far. "edges" reads `client.json`; "no edges" does not, draws inputs
evenly between each feature's lowest and highest edge and fits on raw values.
Agreement is with the server's answers on the whole test split (12211, 7500 and
11303 rows), median over 5 seeds, 20000 queries at most. A constant copy agrees
with the share of the majority answer, 1 - adverse share or the adverse share.

| data | attacker | adverse share | q to 99% | q to 99.9% | q to every test row | agreement at 500 / 1000 / 2000 / 5000 / 20000 queries |
|---|---|---|---|---|---|---|
| Adult | edges | 0.830 | 2381 | 5462 | 5502 | 0.912 / 0.976 / 0.986 / 0.996 / 1.000 |
| Adult | edges + reasons | 0.830 | 2609 | 7477 | 7477 | 0.929 / 0.960 / 0.985 / 0.996 / 1.000 |
| Adult | no edges | 0.830 | > 20000 | > 20000 | > 20000 | 0.830 / 0.874 / 0.925 / 0.984 / 0.987 |
| Taiwan | edges | 0.115 | 1721 | > 20000 | > 20000 | 0.885 / 0.979 / 0.991 / 0.994 / 0.996 |
| Taiwan | edges + reasons | 0.115 | 1649 | > 20000 | > 20000 | 0.885 / 0.979 / 0.992 / 0.995 / 0.997 |
| Taiwan | no edges | 0.115 | > 20000 | > 20000 | > 20000 | 0.885 / 0.874 / 0.631 / 0.581 / 0.597 |
| Bank marketing | edges | 0.939 | 3602 | 14724 | > 20000 | 0.957 / 0.968 / 0.975 / 0.996 / 0.999 |
| Bank marketing | edges + reasons | 0.939 | 2735 | 9793 | 15496 | 0.939 / 0.973 / 0.978 / 0.995 / 1.000 |
| Bank marketing | no edges | 0.939 | > 20000 | > 20000 | > 20000 | 0.939 / 0.680 / 0.892 / 0.942 / 0.961 |

- Approve/decline alone does not keep the model: with the edges, the copy matches
  99% of real decisions after 1600 to 3600 queries, and every test row on Adult
  after about 5500. Returning the score instead gives an exact copy in 1000 to 2500
  (Extraction), so taking the score away costs the attacker a few times more, not more.
- The reasons help it little: about 1.5x fewer queries to 99.9% on Bank marketing,
  nothing on Adult or Taiwan. They only change which features it sweeps, and points
  the copy is unsure about lead to the same thresholds.
- Taiwan's copies stop near 99.6%: some real rows sit where only a few leaf
  combinations decide, and random sweeps rarely reach them.
- Hiding the edges slows this attacker a lot, and on Taiwan its copy is worse than a
  constant one (0.58 to 0.60): inputs drawn evenly between the edges miss where
  Taiwan's skewed amounts lie. We do not count that as a defence. An attacker with
  data from the same population draws where real inputs are, and Edges shows that
  with the prediction back the hidden edges come out bit for bit.
- What the server can still do is count. It sees every request and which
  evaluation key it came with, and a new identity costs a key upload of hundreds of
  MB. The last column is the copy's agreement under such a budget: at 500 queries
  per key it is at the majority share on Taiwan and Adult's attacker reaches 0.91;
  by 2000 every attacker with the edges is at 0.975 or better.

## Published numbers, not measured here

Other work on SHAP for private tree models. Different models, threat models and
hardware, so these are for scale and not a comparison.

| | setting | model | time per explanation | source |
|---|---|---|---|---|
| XorSHAP | semi-honest MPC, full threshold | 60 trees, depth 4, 100 features | 4.5 s (100 samples in 7.5 min, preliminary) | [paper](https://cic.iacr.org/p/1/4/27) |
| CipherExplain | "full-FHE path-product", scheme not named on the page | 100 trees, depth 4, K=40 | about 70 s (vendor figure) | [vendor page](https://vaultbytes.com/cipherexplain) |

The same vendor's [SDK page](https://pypi.org/project/cipherexplain/) (0.8.1) lists
its tree model types as plaintext KernelSHAP with `fhe_mode` ignored, and the
vendor page says hosted CatBoost SHAP is plaintext TreeSHAP. The 70 s path is
not what that SDK runs, and we could not check it.

Nearest row of ours: T=30 at 16 to 35 s on 2 desktop CPUs (see above). We have no
encrypted run at T=100, only Concrete's estimates.
