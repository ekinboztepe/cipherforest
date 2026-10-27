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

## Numbers

Random forest teacher (100 trees) distilled to 3 oblivious trees of depth 6.
Laptop, Concrete 2.11 in a WSL2 VM; details in [experiments/RESULTS.md](experiments/RESULTS.md).

|                              | Adult             | Taiwan credit default |
|------------------------------|-------------------|-----------------------|
| AUC teacher / encrypted model | 0.918 / 0.897    | 0.774 / 0.763         |
| server time per request (median) | 0.69 s        | 0.61 s                |
| client encrypt / decrypt     | 6.7 ms / 1.9 ms   | 7.0 ms / 2.0 ms       |
| evaluation keys (sent once)  | 566 MB            | 335 MB                |
| max error vs plaintext model | 0.005             | 0.004                 |

Every encrypted result was checked against the plaintext integer model: bit-exact.

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
- Client, from queries: a per-feature SHAP vector for every request. Every
  threshold is one of the known edges and the model is a few trees with 2^D
  leaves each, so a client can likely rebuild the whole model with relatively
  few queries. This is not measured yet. Measuring it, and what top-k outputs,
  noise, coarser outputs and data-independent edges cost in explanation quality,
  is the next experiment.

Treat the deployed model as extractable by its clients for now.

## Limits

- The student is small. On the benchmarks here it gives up 0.01 to 0.03 AUC
  against the forest.
- Its SHAP values explain the student exactly, but agree only moderately with
  the teacher's (cosine 0.6 to 0.85).
- Evaluation keys are hundreds of MB per client.

## Layout

```
src/cipherforest/
  oblivious.py    oblivious trees
  shap_table.py   per-leaf exact SHAP tables
  distill.py      greedy distillation into oblivious ensembles
  quantize.py     exact input encoding, sum-preserving output rounding
  circuit.py      Concrete circuit
  deploy.py       client/server artifacts and byte protocol
experiments/      benchmarks and the end-to-end demo
```
