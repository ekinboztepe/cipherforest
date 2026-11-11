# Encrypted tree inference with exact SHAP, and what the explanations give away

Technical report, cipherforest 0.2. Every number below is from
[experiments/RESULTS.md](../experiments/RESULTS.md), which names the script and raw
file behind it. Where something was not measured, the text says so.

## Abstract

A client wants a prediction from a tree model and the reasons for it, without
showing its input to the server; the model owner does not want to hand over the
model. We serve oblivious tree ensembles, the kind CatBoost trains, under TFHE so
that one encrypted request returns the prediction and its exact interventional SHAP
vector. The key fact is that the SHAP vector of an oblivious tree depends on the
input only through the tree's D comparison bits, so it can be tabulated per leaf in
the clear and looked up under encryption. The server runs D encrypted comparisons
and one lookup per feature the tree uses. Input encoding is exact and the only error
is fixed-point rounding of the outputs, bounded by T/(2s) for T trees at scale s.
On five tabular datasets a CatBoost model of 30 trees of depth 4 matches a
100-tree random forest's AUC and is served with about 230 bootstraps per request.
On one rented server, with the same Concrete version and settings, our distilled
models with SHAP ran 3.4 to 4.7 times faster than Concrete-ML's XGBoost without SHAP
on the three largest datasets, and slower on the smallest. We then measure the other
side: a client that can query the server copies a model of this size exactly in
about 700 to 2500 queries, and none of the defences we tried (output noise, coarser
outputs, hidden or uniform quantiser edges, approve/decline answers) raises that
past a few thousand queries at a price an honest client would accept. A per-client
request budget is what remains. To our knowledge this is the first open-source
implementation of encrypted exact SHAP for tree ensembles.

## 1. Setting

Three parties. The model owner trains a model and exports two files: `server.zip`
(the compiled circuit, which contains the thresholds and SHAP tables) and
`client.json` (the input encoder, the scale, the base value and which features the
model uses). The server holds `server.zip` and evaluates the circuit on ciphertexts.
The client holds `client.json` and its secret key, encrypts its input, and decrypts
the answer.

The server is semi-honest: it runs the circuit it was given. It learns that a request
arrived, its size and how long it took, which is the same for every input since the
circuit has no data-dependent branches. It learns nothing about the input, the
prediction or the explanation. The client learns the decrypted answer and what
`client.json` contains. Section 5 measures what it learns from many answers.

TFHE is a fully homomorphic scheme whose programmable bootstrapping (PBS) evaluates
any lookup table on an encrypted small integer while refreshing its noise. Concrete
compiles a Python function on integers into such a circuit. A PBS is the expensive
operation, so we count them.

## 2. Exact SHAP of an oblivious tree is a table

An oblivious tree of depth D tests one pair (feature f_j, threshold t_j) at every
node of level j, the same pair across the level. Its D comparison bits
c_j(x) = [x_{f_j} > t_j] are the binary digits of the leaf index, and the tree
outputs the value of that leaf.

Interventional SHAP explains f(x) against a background set B. The value of a
coalition S of features is v_x(S) = mean over b in B of f(x_S, b_{not S}), the model
with the features outside S replaced by the background row's. The Shapley value
phi_i(x) is the usual weighted average of v_x(S + i) - v_x(S).

**Proposition.** For an oblivious tree, phi(x) depends on x only through
c(x) in {0,1}^D.

*Proof.* In the hybrid input (x_S, b_{not S}), the bit of level j is c_j(x) if f_j is
in S and c_j(b) otherwise. So f(x_S, b_{not S}), and with it v_x(S) for every S, is a
function of c(x) and b only. The Shapley values are fixed combinations of the v_x(S),
so they are a function of c(x) too. Only features the tree tests can have nonzero
values: a feature it never tests does not change any v_x(S).

So a tree has at most 2^D distinct SHAP vectors, one per leaf, over its m <= D
distinct features. We compute them once in the clear, for each leaf by brute force
over the m features against the background. For an ensemble, SHAP values add over
trees. `tests/test_shap_table.py` checks the tables against brute-force permutation
Shapley values on 150 random trees (depth 1 to 5, repeated features): equal to 1e-9.
For an imported CatBoost model the tables agree to 1e-9 with brute-force Shapley
values computed from CatBoost's own predictions, and to 1e-6 with
`shap.TreeExplainer`, which works in float32 there.

## 3. The circuit

**Inputs.** Each feature has a sorted list of edges, and the client encodes x as the
number of edges below it. Every threshold of the model is one of these edges, say
e_k, so x > e_k exactly when the code is above k. Encoding loses nothing for the
model's decisions. For CatBoost the edges are the model's own borders plus 32
training quantiles per feature (so `client.json` does not list only the thresholds);
for distillation the splits are chosen among 32 quantile edges.

**Per tree.** D encrypted comparisons, one PBS each, give the bits; their weighted sum
is the leaf index, with no PBS. Then one PBS per feature the tree uses looks up that
feature's column of the SHAP table at that index, and a clear 0/1 matrix adds the
results into the output vector. So a request costs the sum over trees of
(depth + features used). The server never forms the prediction: the client computes
it as base + sum of SHAP, which keeps prediction and explanation consistent by
construction.

**Outputs.** Table entries are integers at a scale s. Rounding each tree's table so
that every leaf's row keeps its rounded total moves each tree's contribution to the
prediction by at most 1/(2s), so the prediction is off by at most T/(2s); s is the
largest scale for which every partial sum provably fits the output width (8 bits by
default), and the circuit is told that width rather than inferring it from sample
inputs, so no input can wrap around. Two smaller outputs drop the explanation:
"score" looks up each leaf's total once per tree (depth + 1 PBS) and returns the sum;
"decision" also compares it with a clear cut and returns one bit, as an approve or
decline service would.

**Error probability.** Every PBS fails with a small probability. Concrete's default
allows about 1e-5 per lookup; with a few hundred lookups the compiler reports a
chance of a wrong request between 1 in 3400 (3 trees) and 1 in 300 (30 trees), and
we saw 2 wrong answers in 6806 requests on a rented server. With Concrete-ML's
settings (2^-40 per bootstrap, compressed keys and inputs) we saw none in 7500, at
two to three times the server time. These are our defaults.

## 4. Accuracy and speed

**Which model to serve.** Three distilled trees of depth 6 lose 0.01 to 0.03 AUC
against a 100-tree random forest (0.06 on German credit), and a CatBoost model of the
same size trained on the labels does as well, so distillation is not where the value
is. A bigger CatBoost model, served as it is, closes the gap (mean of 3 seeds, AUC of
the integer model the circuit serves):

| data | forest (100 trees) | distilled, 30 x depth 6 | CatBoost, 30 x depth 4 |
|---|---|---|---|
| Adult | 0.918 | 0.905 | 0.921 |
| Taiwan credit default | 0.779 | 0.771 | 0.777 |
| Bank marketing | 0.924 | 0.910 | 0.925 |
| Breast cancer | 0.983 | 0.971 | 0.990 |
| German credit | 0.786 | 0.725 | 0.785 |
| PBS per request (mean) | | 345 | 229 |

So we recommend training CatBoost at depth 4 with about 30 trees and serving it
directly, with its own exact SHAP values. Distillation is for models one is handed
(XGBoost, a forest): a small copy is fitted to the original's scores and its
explanation tables can be fitted to the original's SHAP values, which raises their
cosine agreement with the original from 0.57-0.89 to 0.82-0.95 at 10 trees, at the
cost that they are no longer the copy's own Shapley values.

**Against Concrete-ML.** Concrete-ML's tree models are the closest one-server
alternative; they return the prediction only. On one rented 8 vCPU server (EPYC
9655P), same Concrete version and settings, five datasets and up to five seeds,
our distilled student with SHAP against its XGBoost at the same tree count:

- 3.4x to 4.7x faster on Adult, Taiwan and Bank marketing at 3, 10 and 30 trees
  (at 30 trees: 5.3 to 5.7 s against 18.7 to 23.2 s per request);
- 2.3x to 3.0x on German credit; on Breast cancer 1.6x at 3 trees and 0.7x at 30,
  where Concrete-ML's trees are small and ours are not;
- Concrete-ML's evaluation keys are 3 to 5 times smaller (74 MB against 215 to 396
  MB compressed), and at 30 trees its AUC is higher than the distilled student's.

The model we recommend, CatBoost at depth 4, was timed against Concrete-ML on the
desktop instead: one job at a time in the same container, both pinned to the same 4
cores of a Ryzen 5 3600, seed 0, 20 requests each, median server time per request:

| data, 30 trees | AUC CatBoost / XGB D4 / XGB D6 | CatBoost with SHAP | XGB depth 4 | XGB depth 6 |
|---|---|---|---|---|
| Adult | 0.921 / 0.918 / 0.919 | 4.9 s | 17.6 s | 50.0 s |
| Taiwan credit default | 0.773 / 0.768 / 0.766 | 6.0 s | 17.3 s | 55.5 s |
| Bank marketing | 0.927 / 0.924 / 0.932 | 5.8 s | 17.7 s | 62.0 s |
| German credit | 0.811 / 0.822 / 0.798 | 5.4 s | 15.7 s | 34.5 s |
| Breast cancer | 0.988 / 0.985 / 0.985 | 6.3 s | 9.5 s | 9.9 s |

At 10 trees the picture is the same (2.1 to 2.5 s against 4.7 to 7.6 s and 4.7 to
22.9 s). So, with the explanation, 2.9x to 3.6x faster than Concrete-ML's trees of the
same shape on four datasets and 6x to 11x faster than its depth-6 trees, at similar
AUC; on Breast cancer 1.5x to 1.9x. Without the explanation ours is only 2% to 10%
faster still. The depth-6 ratio is two to three times what the rented server showed
for the distilled student, so these ratios belong to the machine they were measured
on. Concrete-ML keeps its smaller keys (74 MB against 216 to 279 MB) and faster key
generation (2.5 s against 8 to 13 s).

The explanation itself is cheap: on Adult at 30 trees, the SHAP output took 17.1 s
on 4 desktop cores against 13.9 s for the score alone, about a quarter more.

## 5. What a client learns from answers

A client that may query the server can copy the model. The model is small, every
threshold is one of the edges in `client.json`, and the answer is exact, so the
attack is mostly bookkeeping:

1. Sweep each feature through its codes from a starting point; the output changes
   only at a true threshold.
2. Find which features share a tree: with SHAP, phi_f moves when g moves; without,
   a two-feature sweep is not additive.
3. The model is a sum of one table per maximal clique of that graph, so querying a
   grid over each clique and combining by inclusion-exclusion gives it exactly.

The attacker adds a starting point per round until its copy agrees with the server
on 200 fresh random queries, which count. On the deployed 3-tree models (10 seeds):

| attacker | Adult | Taiwan | Bank marketing |
|---|---|---|---|
| prediction only, every cell | 2547 | 28767 | 2962 |
| prediction only, graph | 1382 | 2492 | 963 |
| prediction + SHAP, graph | 687 | 2148 | 864 |

Median queries to a copy that matched the server on 5000 random inputs (four
Bank marketing copies stopped early at 98.7% to 99.7%). The leak comes from the small
model and the public edges, not from SHAP: the explanation saves the attacker 1.1x
to 2x, not the 3x to 13x a naive comparison suggests.

**Defences.** Each was measured from both sides, the honest client's loss and the
attacker's cost, and its cost under FHE:

- *Noise on the outputs* (rounded Laplace, added by the server as a clear integer):
  a naive attacker that repeats inputs pays 10^4 times more at b = 0.5. One that
  fits the clique tables by least squares does far better: at b = 2, where the honest
  client already loses 0.01 AUC, an exact copy costs it 262K to 1M queries, 260 to
  1800 times fewer than repeating, and a copy that agrees on 99% of decisions a few
  thousand. The server cannot spot
  repeats, since every query is a fresh ciphertext. Not a defence.
- *Coarser outputs* (4 to 6 bits): at 4 bits the honest client's explanations drift
  to cosine 0.8 to 0.88 while the attacker pays at most about 5x more.
- *Uniform edges* keep the training quantiles out of `client.json` at little AUC cost,
  but slow the attack by less than 2x.
- *Hidden edges* (the server quantises): the attacker bisects to every edge the model
  uses, bit for bit, at 3x to 7x the queries, while the circuit's wider comparisons
  cost 130x to 200x.
- *Approve/decline answers*, optionally with four reasons on a decline: a learned
  copy agrees with 99% of real decisions after 1600 to 3600 queries, and the reasons
  help it little.

[TO FILL from catboost_attack_bench: the same attacks on CatBoost 10 and 30 x depth 4.]

What the server can still do is count: it sees every request and the evaluation key
it came with, and a new key costs the client an upload of hundreds of MB.
`BudgetedServer` caps the requests per key. At 500 queries per key the approve/decline
attacker's copy is at the majority share on Taiwan; by 2000 every attacker with the
edges agrees with 97.5% or more of real decisions. A budget holds only as well as keys
are tied to real accounts.

## 6. Related work

XorSHAP (Jetchev and Vuille, ePrint 2023/1859) computes exact TreeSHAP for general
tree ensembles under secure multiparty computation, about 4.5 s per explanation for
60 trees of depth 4 and 100 features, amortised over 100 samples; it needs several
non-colluding servers, and we found no public code. Our approach needs one server but
only handles oblivious trees. CipherExplain is a commercial hosted service that
computes SHAP under CKKS; for tree ensembles its public examples compute SHAP on the
client from the model's weights. Private tree inference without explanations is a
larger field: Concrete-ML (TFHE, compared above), RCC-PDTE (BFV), SortingHat (TFHE),
SilentWood (client-aided) and MPC protocols that need several servers. That
explanations help model extraction is known (for example AUTOLYCUS); our measurement
is specific to encrypted serving, where the server cannot see or rate-limit by input.

## 7. Limits

- Oblivious trees only, numeric features without missing values, binary or
  regression outputs. Depth beyond 6 makes the per-leaf tables large; we have not
  split them.
- Each client sends 215 to 396 MB of compressed evaluation keys once, and the server
  holds them expanded, about 1 to 1.5 GB per client.
- All timings are from two kinds of machine (a desktop Ryzen 5 3600 in Docker, and
  rented EPYC containers), and the ratios to Concrete-ML moved between them.
- The attacks are ones we built. A better attacker would only lower the costs.
- Concrete is BSD-3-Clause-Clear and Zama states that commercial use needs its
  patent licence.
- Membership inference from the explanations was not measured.
