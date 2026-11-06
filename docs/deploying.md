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

With the quantiser edges, an approve/decline copy reaches 96% to 98% agreement after
1000 queries and 99% after 1600 to 3600. Pick the budget from that curve, and tie keys
to an account: a fresh key pair is a fresh budget, so the budget only holds as well as
the sign-up does. A key upload is hundreds of MB, which helps but does not stop a
determined attacker.

## 6. Plan memory and bandwidth

- Each client uploads 215 to 396 MB of compressed keys once; the server holds them
  expanded, about 1 to 1.5 GB per client (2 to 2.7 GB for the decision output). Load
  them once per client, not per request.
- A request is under 1 KB in, a few hundred bytes out.

## 7. Install in one go

Concrete needs numpy below 2 and, in fresh environments, setuptools. Install
everything in one command so the resolver respects that:

```
pip install "cipherforest[fhe,catboost]"
```

Adding CatBoost, pandas or similar later on its own can pull numpy 2 and break the
encrypted part. Concrete has no Windows wheels; use `docker/Dockerfile` there.

## 8. Know what is not covered

A malicious server, side channels beyond timing and access patterns, membership
inference from SHAP values, and commercial use of Concrete without Zama's patent
licence. See SECURITY.md.
