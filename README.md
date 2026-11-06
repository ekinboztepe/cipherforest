# cipherforest

Encrypted inference for tree models that returns the prediction *and* its exact
SHAP explanation. The server never sees the input, the prediction or the explanation.

## How it works

1. Take an ensemble of oblivious trees, as CatBoost trains them, or distil another
   model into a few: every level tests one (feature, threshold) pair, so the D
   comparison bits of a sample are its leaf index.
2. Interventional SHAP of an oblivious tree depends on the input only through
   those D bits. The full SHAP vector of every leaf is computed once, in the clear.
3. Under TFHE the server runs D encrypted comparisons and one lookup per output.
   The client decrypts the SHAP vector and gets the prediction as `base + sum(SHAP)`.

Splits are restricted to the quantiser edges, so encoding the inputs as integers
is exact. The only error is fixed-point rounding of the outputs, bounded by
`trees / (2 * scale)`.

## Use

```
pip install "cipherforest[catboost]"        # model preparation, any OS
pip install "cipherforest[catboost,fhe]"    # plus the encrypted parts, Linux or macOS
```

Not on PyPI yet: until then, `pip install "cipherforest[catboost] @ git+https://github.com/ekinboztepe/cipherforest"`.
Concrete needs numpy below 2. Install everything in one command, as above; adding
CatBoost or pandas later on its own can pull numpy 2 and break the encrypted part.

A CatBoost model with symmetric trees (CatBoost's default), one output and numeric
features is served as it is:

```python
import cipherforest

model = cipherforest.from_catboost(cb_model, background=x_train[:100], x_train=x_train)
score, shap = cipherforest.explain(model, x_test)       # in the clear, what clients will decrypt
cipherforest.export(cipherforest.compile_model(model), model, Path("artifacts"))
```

`artifacts/server.zip` goes to the server, `artifacts/client.json` to clients.
Then, in bytes over any transport:

```python
client = cipherforest.Client.load(Path("client.json"))
keys = server.load_keys(client.evaluation_keys())        # once per client, hundreds of MB
pred, shap = client.decrypt(server.run(client.encrypt(x), keys))
```

Any other model (XGBoost, LightGBM, a forest) goes through a small oblivious copy
fitted to its scores; the copy is what gets explained, exactly:

```python
model = cipherforest.distill(x_train, teacher.predict_proba(x_train)[:, 1], depth=6, n_trees=3)
```

`x_train` adds training quantiles to the edges, so `client.json` does not list only
the model's own thresholds. Imported SHAP values match brute-force Shapley values
computed from CatBoost's own predictions to 1e-9, and `shap.TreeExplainer` to 1e-6
(shap works in float32 there). A full run is in [examples/quickstart.py](examples/quickstart.py).

### Three outputs

| `compile_model(..., output=)` | client gets | bootstraps per tree |
|---|---|---|
| `"shap"` (default) | prediction and exact SHAP vector | depth + features the tree uses |
| `"score"` | prediction only | depth + 1 |
| `"decision"`, with `cut=model.cut_for(t)` | one bit: prediction > t | depth + 1, plus 1 overall |

The decision output is an approve/decline API: the server compares on ciphertexts
and the client learns one bit. Measured on Adult at 30 trees, the explanation adds
about a quarter to the server time of the prediction alone (17.1 s against 13.9 s
on the desktop), and the decision costs the same as the score.

## Numbers

Random forest teacher (100 trees) distilled to 3 oblivious trees of depth 6.
Desktop (Ryzen 5 3600, 16 GB), Concrete 2.11 in a WSL2 VM; details in [experiments/RESULTS.md](experiments/RESULTS.md).
Measured with Concrete's default settings, which are no longer ours (see Limits): the
current default is 2 to 3 times slower and sends smaller keys.

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

Treat the deployed model as extractable by its clients. An approve/decline server
slows copying to a few thousand queries but does not stop it, and `BudgetedServer`
caps the requests per set of evaluation keys. [docs/deploying.md](docs/deploying.md)
walks through choosing the output, the budget and the settings.

## Limits

- A distilled student is small. On the benchmarks here it gives up 0.005 to 0.06 AUC
  against a 100-tree forest. CatBoost trained directly does not: at depth 4 with 30
  trees it matches the forest on all five datasets, with its own exact SHAP values and
  a cheaper circuit (about 230 bootstraps against 340). Distil only models you are
  handed, not ones you train.
- Its SHAP values explain the student exactly, but agree only moderately with
  the teacher's (cosine 0.6 to 0.85). `distill(..., teacher_shap=(rows, values))`
  refits the tables to the teacher's values instead, prediction and circuit cost
  unchanged: cosine 0.82 to 0.95 at 10 trees. They are then no longer the student's
  own Shapley values.
- Each client sends 215 to 396 MB of compressed evaluation keys, and the server
  holds them expanded, about 1 to 1.5 GB per client.
- With Concrete's default settings a request can decrypt to a wrong answer: the
  compiler reports a chance of about 1 in 3400 at 3 trees and 1 in 300 at 30, and we
  saw 2 in 6806 requests. `compile_model` therefore uses Concrete-ML's settings
  (`p_error=2**-40`, compressed keys and inputs) unless given a `configuration`:
  none wrong in 7500 requests, at 2 to 3 times the server time.
  `fhe.Configuration()` gives Concrete's defaults back.
- Concrete is BSD-3-Clause-Clear, which grants no patents, and Zama states that
  commercial use needs its patent licence. Fine for research, not for a product as is.

## Related work

The closest work we know is XorSHAP (Jetchev and Vuille, [ePrint 2023/1859](https://eprint.iacr.org/2023/1859)):
exact TreeSHAP for ordinary tree ensembles under secure multiparty computation, about
4.5 s per prediction (100 at once, 60 trees of depth 4, 100 features). It needs several
non-colluding servers; this needs one, but only works on oblivious trees. We found no
public code for it.

CipherExplain ([VaultBytes](https://github.com/VaultBytes/cipherexplain-examples), 2026) is a
hosted API that computes SHAP under CKKS. For tree ensembles its examples route leaves
under encryption and compute SHAP on the client from the model's own weights, so the
client owns the model; its server code is not public.

Private tree evaluation without explanations is a larger field: Concrete-ML (TFHE,
one server, the comparison above), RCC-PDTE ([code](https://github.com/RasoulAM/private-decision-tree-evaluation),
BFV, one server), SortingHat (TFHE), SilentWood (BFV, client-aided, no public code)
and MPC protocols such as Mostree and Kangaroo that need several servers.

See [SECURITY.md](SECURITY.md) for the threat model, parameters and failure probability.

## Layout

```
src/cipherforest/
  api.py            from_catboost, distill, explain
  oblivious.py      oblivious trees
  shap_table.py     per-leaf exact SHAP tables
  distill.py        greedy distillation into oblivious ensembles
  catboost_json.py  CatBoost import
  quantize.py       exact input encoding, sum-preserving output rounding
  circuit.py        Concrete circuit
  deploy.py         client/server artifacts and byte protocol
examples/           quickstart
experiments/        benchmarks and the end-to-end demo
```
