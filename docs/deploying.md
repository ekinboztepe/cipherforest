# Deploying cipherforest

A checklist for putting a model behind an encrypted endpoint. The numbers come from
[experiments/RESULTS.md](../experiments/RESULTS.md); the threat model is in
[SECURITY.md](../SECURITY.md).

## 1. Get an oblivious model

- **CatBoost** (symmetric trees, the default): `cipherforest.from_catboost(model, background, x_train)`.
  Served as it is, no accuracy lost beyond fixed-point rounding of the outputs.
- **Anything else**: `cipherforest.distill(x_train, model.predict_proba(x_train)[:, 1])`.
  A small oblivious copy. On the benchmarks here 3 to 10 trees of depth 6 lose 0.01 to
  0.06 AUC against a 100-tree forest. Add `teacher_shap=(rows, shap_values)` if users
  should see the original model's reasons rather than the copy's own; the prediction
  stays the same.

Check it in the clear with `cipherforest.explain(model, x)`: that is bit for bit what
the encrypted path returns.

## 2. Size it

Each tree costs `depth` bootstraps for its comparisons, plus one lookup per feature it
uses (SHAP output) or one lookup in all (score and decision outputs). Server time per
request, 8 vCPU EPYC 9655P, SHAP output: about 0.7 s at 3 trees, 1.9 s at 10, 5.5 s at
30. More trees past 10 rarely bought AUC here.

## 3. Choose the output

| output | when |
|---|---|
| `"shap"` | the client should see why: the prediction and every feature's share of it |
| `"score"` | the prediction only; about 20% faster than SHAP |
| `"decision"`, `cut=model.cut_for(threshold)` | approve/decline; the client learns one bit |

Taking the explanation or the score away does not stop a client from copying the
model; it makes copying a few times dearer (RESULTS.md, Label-only answers).

## 4. Keep the safe settings

`compile_model` uses `cipherforest.SAFE` (p_error 2^-40 per bootstrap). Concrete's own
default decrypts wrong about once in 300 to 3000 requests. Do not pass a configuration
unless you mean to trade that.

## 5. Budget each client

The server cannot see inputs or answers, but it can count requests per set of
evaluation keys:

```python
server = cipherforest.BudgetedServer(Path("server.zip"), budget=1000)
keys = server.load_keys(uploaded_key_bytes)
server.run(ciphertext, keys)   # raises BudgetExceeded past 1000 for these keys
```

With the quantiser edges, an approve/decline copy of a 3-tree model reaches 96% to 98%
agreement after 1000 queries and 99% after 1600 to 3600; for CatBoost depth 4 x 30 it is
about 97% after 20000 (RESULTS.md, "Copying the CatBoost model"). Pick the budget from
those curves, and tie keys to an account: a fresh key pair is a fresh budget, so the
budget only holds as well as the sign-up does. A key upload is hundreds of MB, which
helps but does not stop a determined attacker.

Tying keys to accounts tells the server whose keys are whose. Where that matters, the
issuance can be anonymous: the account holder gets a fixed number of key-registration
tokens per period from an issuer that knows who they are, and redeems one with each key
upload. With anonymous rate-limited credentials (ARC, an IETF Privacy Pass draft) the
server can check that a token is valid and unspent but not which account it came from,
so it limits requests per account (tokens times budget) without learning which keys
belong to whom. This is a design, not code in this package: ARC's draft has expired
and we know of no maintained Python implementation.

## 6. Plan memory and bandwidth

- Each client uploads 215 to 396 MB of compressed keys once; the server holds them
  expanded, about 1 to 1.5 GB per client (2 to 2.7 GB for the decision output). Load
  them once per client, not per request.
- A request is under 1 KB in, a few hundred bytes out.
- One request spreads over the cores by itself: CatBoost depth 4 x 30 took 13 s on one
  logical CPU of a desktop, 7.7 s on two, 5.6 s on four and no less on six. Beyond
  that, add workers, each with its own copy of the clients' keys in memory.

## 7. Retrain without new keys

A retrained model would normally need new keys from every client. Compile it for the
keys already deployed instead, and keep the client's secret keys between versions:

```python
compiled = cipherforest.compile_model(new_model, same_keys_as=Path("v1/client.json"))
cipherforest.export(compiled, new_model, Path("v2"))

# client: keep v1's keys (they hold the secret key, never send them)
stored = old_client.secret_keys()
client = cipherforest.Client.load(Path("v2/client.json"), keys=stored)
# server: v2/server.zip answers with the evaluation keys it already holds for this client
```

On Adult this worked for a model retrained on other data with the same shape, for one
with twice the trees and for the score-only output, every answer exact and no slower
than the same model compiled freely. A model that needs other parameters (deeper
trees, or categorical features where there were none) is refused with a ValueError,
and its clients must send new keys. Keep the shape fixed between versions to stay
within the keys.

## 8. Install in one go

Concrete needs numpy below 2 and, in fresh environments, setuptools. Install
everything in one command so the resolver respects that:

```
pip install "cipherforest[fhe,catboost]"
```

Adding CatBoost, pandas or similar later on its own can pull numpy 2 and break the
encrypted part. Concrete has no Windows wheels; use `docker/Dockerfile` there.

## 9. Know what is not covered

A malicious server, side channels beyond timing and access patterns, membership
inference from SHAP values, and commercial use of Concrete without Zama's patent
licence. See SECURITY.md.
