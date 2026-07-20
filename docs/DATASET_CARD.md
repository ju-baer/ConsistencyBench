# Dataset Card: ConsistencyBench

This card documents the dataset artifacts produced by this repository, in
the format expected for HuggingFace Hub dataset cards. The machine-generated
versions of this card (one per HF dataset repo) are written by
`src/consistencybench/analysis/hf_export.py::write_dataset_cards` into
`data/hf_export/README_{probes,results,interpretability}.md` after a full
pipeline run, ready to `huggingface-cli upload` alongside their CSVs.

## Dataset Summary

ConsistencyBench is a probe-pair dataset for evaluating **logical
consistency** in LLMs — whether a model gives compatible answers to two
prompts that are formally guaranteed, by a truth-preserving transformation,
to require compatible answers. It is explicitly **not** an accuracy
benchmark: probes are scored on internal coherence, not correctness.

Three related dataset bundles are produced:

| Bundle | Rows (reference run) | Model-agnostic? |
|---|---|---|
| `consistencybench-probes` | 4,500 probe pairs | Yes (prompts only) |
| `consistencybench-results` | 4,500 x 16 scored (probe, model) pairs | No (includes model responses/scores) |
| `consistencybench-interpretability` | layer probing / patching / steering bundle | No (single local model) |

## Supported Tasks

- **LLM consistency evaluation** — score a new model's responses against the
  probe set with `src/consistencybench/scoring.py` and compare against the
  leaderboard.
- **Calibration research** — the results bundle includes per-response LJS
  judge confidence, sufficient to recompute CCS for any subset of models.
- **Mechanistic interpretability** — the interpretability bundle is a
  ready-made labeled activation dataset (`inconsistent` vs. `consistent`)
  for probing/steering research on open-weight models.

## Languages

English (`en`) only in the current release.

## Dataset Structure

### `consistencybench-probes` schema

| Field | Type | Description |
|---|---|---|
| `probe_id` | int | Unique probe identifier |
| `transformation_family` | str | One of `composition, reversal, complement, ordering, equivalence` |
| `domain` | str | One of `general, science, ethics` |
| `difficulty` | str | One of `easy, medium, hard` (derived from `semantic_distance_delta`, not self-reported) |
| `semantic_distance_delta` | float | delta in [0,1], see [METHODOLOGY.md](METHODOLOGY.md#3-the-difficulty-metric-δ-semantic-distance) |
| `prompt_a`, `prompt_b` | str | The two prompts in the pair |
| `logical_constraint` | str | Human-readable statement of the formal constraint linking them |
| `expected_inconsistency` | str | What an inconsistent response pair would look like |
| `scoring_hint` | str | `ljs` or `rbs` — which scoring method applies |
| `difficulty_rationale` | str | Generator model's stated reasoning for the difficulty band (informational; not used by the pipeline itself) |

### `consistencybench-results` schema (adds to the above)

`model, model_id, response_a, response_b, consistent, inconsistent,
score_method, score_reason, ljs_confidence, ans_a_summary, ans_b_summary`

Plus a separate `consistencybench_leaderboard.csv`: one row per model with
`overall_ir, ir_<family>, ir_<difficulty>, ir_<domain>, cp_<family>, ccs,
hint_induced_ir_pct`.

### `consistencybench-interpretability` contents

`layer_probe_results.csv, activation_patching.csv,
steering_dose_response.csv, hint_sensitivity.csv, metadata.json` — see
[INTERPRETABILITY.md](INTERPRETABILITY.md) for what each column means.

## Dataset Creation

**Curation rationale:** hand-authoring thousands of probe pairs with a
verified logical relationship does not scale; a single unconstrained
generation call produces prompts that only loosely honor the intended
relationship. See [METHODOLOGY.md](METHODOLOGY.md#2-probegen-4-stage-constraint-driven-probe-synthesis)
for the 4-stage constraint-driven pipeline used instead.

**Source data:** probes are synthetically generated (Gemini 2.5 Pro as the
generator model, `config.GENERATOR_MODEL`), not derived from any existing
benchmark or copyrighted text corpus, over three general-knowledge-style
domains (general, science, ethics). They are not sourced from real user
data and contain no personal information.

**Annotations:** ground truth for `easy`/`medium`/`hard` is a deterministic
function of measured semantic distance (embeddings + lexical/syntactic
distance — a reproducible calculation, not a subjective label). "Correct"
consistency verdicts on model responses come from the RBS+LJS scoring
ensemble; see the cross-judge validation and inter-annotator agreement
methodology in [METHODOLOGY.md](METHODOLOGY.md#5-scoring-rbs--ljs-ensemble)
for the checks run on scoring quality itself.

## Known Limitations

- The five transformation families are documented to cover roughly 83% of
  naturally occurring logical relationships in QA — multi-hop reasoning,
  modal operators, and quantifier scope are out of scope for this release
  (see [TRANSFORMATION_FAMILIES.md](TRANSFORMATION_FAMILIES.md#design-rationale-why-these-five-and-not-others)).
- LJS (LLM-as-Judge) scoring for Composition and Equivalence inherits
  whatever biases the judge model has; cross-judge and RBS-agreement checks
  bound but do not eliminate this.
- English-only.
- The interpretability bundle reflects a single, comparatively small
  open-weight model and should not be assumed to generalize to the larger,
  closed-weight models in the results bundle.

## Licensing

CC-BY-4.0 for all three dataset bundles. Code that generates and evaluates
the datasets is MIT-licensed — see [`LICENSE`](../LICENSE).

## Citation

See the [README](../README.md#citation) for the citation entry.
