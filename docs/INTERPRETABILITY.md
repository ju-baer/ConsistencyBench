# Interpretability Extension (Part B)

Part A treats every model as a black box: prompts in, text out, scored
behaviorally. Part B asks a different question of a single **local,
open-weight model** (`Qwen2.5-1.5B-Instruct` by default,
`config.INTERP_MODEL_ID`) where full white-box access is possible: **where,
mechanistically, does inconsistency come from, and can it be causally
controlled?**

This model is a **dedicated interpretability testbed**, not a leaderboard
entry — it is far smaller than the 16 API models evaluated in Part A and its
own IR number is reported only as a reference point
(`interpretability/activations.py::score_local_model`), never compared
head-to-head against the frontier models.

Everything in this document is implemented in
`src/consistencybench/interpretability/`.

## Why not just use a pretrained SAE?

The obvious tool for this kind of question is a Sparse Autoencoder (SAE)
trained on the model's activations — but no pretrained SAE exists for this
exact checkpoint. Rather than train one from scratch (expensive, and its own
research project), this extension uses two lighter-weight, always-available
substitutes, documented as such throughout the code and this doc:

1. **Linear probe directions** as the intervention vector, following the
   diff-in-means / linear-direction methodology from activation-steering
   work (Turner et al., 2023) — used in place of an SAE feature direction.
2. **PCA components** of the best layer's activations as "pseudo-features",
   checked for consistent/inconsistent separation — analogous in spirit to
   inspecting individual SAE feature directions, but computed on the fly
   from an orthogonal basis rather than a learned overcomplete dictionary.

Neither is a substitute for a real trained SAE if one becomes available for
this checkpoint; both are explicit, labeled approximations.

## Stage 1 — Activation extraction (`activations.py`)

`LocalHFBackend.query_with_activations` runs the model exactly as in Part A
(same system prompt, same greedy/`temperature=0` decoding) but additionally
captures the **full residual stream** — every layer's hidden state at the
final prompt token, the representation the model is actually conditioning
its answer on — for both `prompt_a` and `prompt_b` of a balanced sample of
`config.INTERP_N_PROBES` probes (evenly split across the five transformation
families, `sample_balanced_probes`).

Each probe's activations are saved to `{probe_id}_a.npy` /
`{probe_id}_b.npy`, shape `[n_layers+1, hidden_size]` (index 0 = embedding
output, indices 1..n = each decoder layer's output). Responses are scored
with the same RBS+LJS ensemble as every other model
(`activations.score_local_model`), and `build_activation_matrix` assembles
the final labeled matrix: `X_by_layer[l]` of shape `[n_examples,
hidden_size]`, `y` (1 = inconsistent), aligned by probe ID.

## Stage 2 — Shortcut probing (`shortcut_probe.py`)

**Core question:** is inconsistency linearly decodable from activations
*before generation finishes* — evidence that the model relies on a
consistent internal "shortcut" representation that happens to be wrong,
rather than inconsistency being unstructured, unpredictable noise?

- `train_layer_probes` trains a `LogisticRegression` probe at **every
  layer**, 5-fold stratified CV, `StandardScaler`-normalized, reporting test
  accuracy and AUROC against a majority-class chance baseline. The layer
  with peak test accuracy is `best_layer`; its fitted weight vector is saved
  as the "shortcut direction" for Stage 3.
- `family_conditioned_decodability` repeats the probe *within* each
  transformation family at `best_layer`: is inconsistency equally
  predictable for every family, or are some families' failures more
  "shortcut-like" (early, linearly obvious) than others (only decodable
  from accumulated computation near the output)?
- `pca_pseudo_features` extracts the top 20 principal components at
  `best_layer` and ranks them by Cohen's-d separation between consistent and
  inconsistent examples — the SAE-substitute described above.

All outputs are saved under `data/interpretability/`:
`layer_probe_results.csv`, `family_probe_results.csv`,
`pca_pseudo_features.csv`, `probe_directions.npz`.

## Stage 3 — Causal tests (`causal_tests.py`)

Everything in Stage 2 is **correlational** — decodability doesn't prove the
activation *causes* the inconsistent output. Stage 3 runs two genuine
interventions to close that gap.

### Test 1: Activation patching

For each transformation family, take one probe the model was **consistent**
on and one it was **inconsistent** on. Literally overwrite the
inconsistent-run's residual stream at `best_layer`'s last token position
with the *consistent* run's activation at the same layer/position
(`make_overwrite_hook`), then let generation continue and check the result:

- If the patched output flips to match the consistent run's answer
  orientation (`flipped_toward_source`), the activation at that layer isn't
  just correlated with the outcome — it is (at least partially) causally
  responsible for it.
- Reported per-family and as an overall flip rate; saved to
  `activation_patching_results.csv`.

### Test 2: Feature steering (diff-in-means dose-response)

- The intervention direction is the (unit-normalized) difference between
  the mean consistent-example activation and the mean inconsistent-example
  activation at `best_layer` (`X_by_layer[best_layer][y==0].mean(axis=0) -
  X_by_layer[best_layer][y==1].mean(axis=0)`).
- This direction, scaled by `alpha in {0, 4, 8, 16}`, is **added** (not
  overwritten — `make_add_hook`) to every token position's residual stream
  via a forward hook on the target decoder layer, for a held-out set of up
  to 30 originally-inconsistent probes.
- At each `alpha`, the steered `prompt_b` response is re-scored against the
  (unsteered) `prompt_a` response, and the resulting IR is recorded — a
  genuine **dose-response curve**: does turning up the steering strength
  monotonically reduce inconsistency? Saved to `steering_dose_response.csv`.

### Specificity check

The strongest steering vector (`alpha=16`) is applied to four unrelated
factual control questions (capital of Japan, spider legs, WWII end year,
gold's chemical symbol). If those answers remain coherent and correct, the
intervention is targeted at consistency behavior specifically, rather than
just degrading the model's outputs indiscriminately — ruling out the trivial
explanation that steering "works" only by making the model incoherent.

## Interpreting `best_layer` in the code

`hidden_states[0]` is the embedding output; `hidden_states[l]` for `l >= 1`
is the output of decoder layer `l-1`. So "intervening at
`hidden_states[best_layer]`" (the probe's best layer) means hooking the
**output of decoder layer `best_layer - 1`** — this index shift is handled
once, centrally, in `causal_tests.get_target_module`, so every downstream
function can talk about layers in the same `hidden_states`-indexed numbering
the probes were trained on.

## Running Part B

```bash
python scripts/05_run_interpretability.py
```

Requires a GPU for reasonable runtime (a free-tier Colab T4 works; CPU will
run but slowly, given `INTERP_N_PROBES x 2` forward-generation passes plus
the steering sweep). Needs `pip install -e ".[interpretability]"` for the
extra `torch`/`transformers`/`accelerate`/`bitsandbytes` dependencies (kept
optional so Part A can run in a lighter environment).

## What this extension does and doesn't claim

- It **does** show that inconsistency has a locatable, linear signature in
  this specific 1.5B model's residual stream, and that intervening on that
  signature causally shifts the model's outputs — a genuine mechanistic
  result on the model tested.
- It does **not** claim this generalizes to the 16 much larger, closed-weight
  API models in Part A, several orders of magnitude bigger with unknown
  architectures — no white-box access to those exists. Part B is a
  proof-of-concept methodology on an accessible model, offered as a
  template that could, in principle, be applied to any open-weight model
  under evaluation.
- The PCA/diff-in-means substitutes for a real SAE are explicit
  approximations, not a claim of feature-level, monosemantic
  interpretability in the strict SAE-research sense.
