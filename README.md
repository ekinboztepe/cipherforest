# cipherforest

Encrypted inference for tree models that returns the prediction *and* its exact
SHAP explanation. The server never sees the input, the prediction or the explanation.

## How it works

1. Distil a tree ensemble into a few oblivious trees: every level tests one
   (feature, threshold) pair, so the D comparison bits of a sample are its leaf index.
2. Interventional SHAP of an oblivious tree depends on the input only through
   those D bits. The full SHAP vector of every leaf is computed once, in the clear.
3. Under TFHE the server runs D encrypted comparisons and one lookup per output.
   The client decrypts the SHAP vector and gets the prediction as `base + sum(SHAP)`.

Splits are restricted to the quantiser edges, so encoding the inputs as integers
is exact. The only error is fixed-point rounding of the outputs, bounded by
`trees / (2 * scale)`.

```python
quantizer = FeatureQuantizer.fit(x_train, n_bins=32)
student = distill_ensemble(x_train, teacher_proba, depth=6, n_trees=3, thresholds=quantizer.edges)
tables = [build_shap_table(tree, background) for tree in student.trees]
model = quantize_model(student, tables, quantizer, output_bits=8)
export(compile_model(model), model, Path("artifacts"))  # server.zip + client.json
```

A CatBoost model with symmetric trees, one output and numeric features can be
served directly, no distillation:

```python
cb_model.save_model("model.json", format="json")
padding = FeatureQuantizer.fit(x_train, n_bins=32)
ensemble, quantizer = load_catboost_json(Path("model.json").read_text(), extra_edges=padding)
```

The JSON only keeps the borders the trees use. Without `extra_edges` they become
the quantiser edges, and `client.json` then lists every threshold of the model.
Imported SHAP values match brute-force Shapley values computed from CatBoost's
own predictions to 1e-9, and `shap.TreeExplainer` to 1e-6 (shap works in
float32 there).

## Numbers

Random forest teacher (100 trees) distilled to 3 oblivious trees of depth 6.
Desktop (Ryzen 5 3600, 16 GB), Concrete 2.11 in a WSL2 VM; details in [experiments/RESULTS.md](experiments/RESULTS.md).

|                              | Adult             | Taiwan credit default |
|------------------------------|-------------------|-----------------------|
| AUC teacher / encrypted model | 0.918 / 0.897    | 0.774 / 0.763         |
| server time per request (median) | 0.69 s        | 0.61 s                |
| client encrypt / decrypt     | 6.7 ms / 1.9 ms   | 7.0 ms / 2.0 ms       |
| evaluation keys (sent once)  | 566 MB            | 335 MB                |
| max error vs plaintext model | 0.005             | 0.004                 |

Every encrypted result was checked against the plaintext integer model: bit-exact.

Against Concrete-ML's own XGBoost, on one rented 8 vCPU server (EPYC 9655P) with the
same Concrete version and settings, over five datasets and up to five seeds, at 3 to
30 trees:

- 3.4 to 4.7 times faster on the three largest datasets, 2.3 to 3 times on German credit.
- On Breast cancer the gap closes: 1.6 times at 3 trees, and at 30 trees Concrete-ML
  is the faster one (0.7).
- This also returns SHAP. Concrete-ML's trees are prediction only, but its keys are
  3 to 5 times smaller, and at 30 trees its AUC is higher on all five, by 0.005 to 0.057.

Those ratios hold for that machine only; see the tables in
[experiments/RESULTS.md](experiments/RESULTS.md).

## Running

```
uv sync
uv run pytest                                        # plaintext parts
docker build -f docker/Dockerfile -t cipherforest .
docker run --rm cipherforest                         # all tests, including encrypted ones
```

Concrete has no Windows wheels, hence the image.

## What leaks

- Server: nothing about inputs or outputs. It only handles ciphertexts and evaluation keys.
- Client, from `client.json`: the quantiser edges, which features the model uses,
  and the base value. The edges are per-feature quantiles of the training data,
  so they reveal its marginal distributions.
- Client, from queries: the whole model. Every threshold is one of the known
  edges and the model is a few trees with 2^D leaves each, so a client copies
  it, prediction and SHAP, in about 700 to 2200 queries on the benchmarks here;
  about 1000 to 2500 without SHAP outputs. The copies are exact, apart from some
  that stop early and get about 1% of inputs wrong
  ([experiments/RESULTS.md](experiments/RESULTS.md)). None of the defences we
  measured stops this. Noise on the outputs is averaged away by least squares.
  Hiding the edges behind server-side quantisation costs the client a few
  thousand more queries, and it recovers every edge the model uses bit for bit,
  while the wider comparisons make the circuit 130 to 200 times as expensive.
  Evenly spaced edges keep the training quantiles private but not the model.
  Coarser outputs cost the honest client explanation quality long before they
  cost the attacker more than a few thousand queries.

Treat the deployed model as extractable by its clients.

## Limits

- The student is small. On the benchmarks here it gives up 0.01 to 0.03 AUC
  against the forest.
- Its SHAP values explain the student exactly, but agree only moderately with
  the teacher's (cosine 0.6 to 0.85).
- Evaluation keys are hundreds of MB per client.
- With Concrete's default settings a request can decrypt to a wrong answer: the
  compiler reports a chance of about 1 in 3400 at 3 trees and 1 in 300 at 30, and we
  saw 2 in 6806 requests. With `p_error=2**-40` we saw none in 7500, at 2 to 3 times
  the server time. Pass that `configuration` to `compile_model` for real use.

## Layout

```
src/cipherforest/
  oblivious.py      oblivious trees
  shap_table.py     per-leaf exact SHAP tables
  distill.py        greedy distillation into oblivious ensembles
  catboost_json.py  CatBoost import
  quantize.py       exact input encoding, sum-preserving output rounding
  circuit.py        Concrete circuit
  deploy.py         client/server artifacts and byte protocol
experiments/        benchmarks and the end-to-end demo
```
