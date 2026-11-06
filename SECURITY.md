# Security model

What cipherforest protects, under which assumptions, and what it does not. The
numbers come from [experiments/RESULTS.md](experiments/RESULTS.md).

## Parties

- **Model owner** trains the model and exports `server.zip` and `client.json`.
- **Server** holds `server.zip`, which contains the thresholds and SHAP tables, and
  runs the circuit on ciphertexts.
- **Client** holds `client.json` and the secret key, encrypts its input, and
  decrypts the result.

## What each party learns

| | learns | does not learn |
|---|---|---|
| Server | that a request came, its size, the time it took (the same for every input), the evaluation keys | the input, the prediction, the explanation |
| Client | the decrypted output; from `client.json` the quantiser edges, the features the model uses, the scale and base value | the thresholds and tables, directly |

The edges are per-feature quantiles of the training data unless the owner chose
otherwise, so `client.json` reveals the marginal distribution of each feature.

**The client learns the model from its answers.** With the edges known, a copy of
the prediction and SHAP outputs takes about 700 to 2200 queries, and of the
prediction alone about 1000 to 2500. An approve/decline answer slows this down
but does not stop it (see "Label-only answers" in RESULTS.md). None of the
defences we measured stops extraction at an acceptable cost. Deploy as if every
client can copy the model; the server can count requests per evaluation key,
and that budget is the lever that remains.

## Assumptions

- **Semi-honest server.** It runs the circuit it was given. A server that runs a
  different circuit can return wrong answers, and nothing here detects that.
- **The client keeps its secret key and does not report decryptions to the server.**
  TFHE decryption can fail with small probability, and an attacker who sees which
  decryptions failed learns about the key (IND-CPA^D attacks). Here only the client
  decrypts, so the server never sees a failure. If a deployment sends decrypted
  answers back to the server, treat that as a different threat model.
- **Concrete's parameters are sound.** Concrete picks the lattice parameters for
  128-bit security. We use them as they come and have not re-estimated them.

## Failure probability

Every programmable bootstrap can return a wrong value with small probability.
`compile_model` uses `cipherforest.SAFE` by default: `p_error = 2^-40` per bootstrap,
as in Concrete-ML. With Concrete's own default (about 1e-5 per lookup) a request
of a few hundred lookups decrypts wrong about once in 300 to 3000 times, and we
saw 2 wrong answers in 6806 requests. With `SAFE` we saw none in 7500.

To read the real figure for a compiled circuit:

```python
compiled = compile_model(model)
feedback = compiled.circuit.server._library.get_program_compilation_feedback()
print(feedback.global_p_error)  # chance that one request is wrong
```

In Concrete 2.10 `circuit.global_p_error` returns the per-lookup value, not this one.

## Side channels

- **Timing.** The circuit has no data-dependent branches: the same comparisons and
  lookups run for every input, so server time does not depend on the input. It does
  vary between runs and machines (see RESULTS.md).
- **Memory access.** Each lookup is a blind rotation over the whole table, not an
  indexed read, so which leaf an input reaches does not change the access pattern.
- **Output width.** Outputs are hinted to the width the model guarantees, so no
  input can wrap around a narrower width chosen from the inputset.

## Out of scope

- A malicious server, and checking that it ran the right circuit.
- Fault injection and other physical attacks on the server.
- Membership inference from SHAP values; published attacks exist and were not
  measured here.
- Anything Concrete itself gets wrong.

## Licence of the backend

Concrete is BSD-3-Clause-Clear, which grants no patent rights, and Zama states
that commercial use needs its patent licence. cipherforest itself is Apache-2.0.

## Reporting

Please open an issue on GitHub. For anything that should not be public first, say
so in the issue without details and we will arrange another channel.
