# Methodology

This document is the detailed methodological reference for ConsistencyBench.
For a quick overview, see the [README](../README.md); for the five
transformation families specifically, see
[TRANSFORMATION_FAMILIES.md](TRANSFORMATION_FAMILIES.md); for the
interpretability extension, see [INTERPRETABILITY.md](INTERPRETABILITY.md).

## 1. The core object: a probe pair

Every unit of evaluation is a **probe pair** `(prompt_a, prompt_b)` built so
that a logically coherent agent is *formally required* to answer them in a
related way — either the *same* way (e.g. reversal, ordering) or in a
*compatible* way that a judge can assess (e.g. composition, equivalence), or
must *not* simultaneously affirm both (complement). Each pair carries:

```
probe_id, transformation_family, domain, difficulty, semantic_distance_delta,
prompt_a, prompt_b, logical_constraint, expected_inconsistency, scoring_hint,
difficulty_rationale
```

A model is **consistent** on a probe if its two independent responses satisfy
the family's constraint; **inconsistent** otherwise. The dataset never scores
factual accuracy — a model can be consistently *wrong* (e.g. confidently
affirm a false claim in both prompts) and score as consistent. This is
deliberate: ConsistencyBench measures a different failure mode than standard
accuracy benchmarks (see the "Two independent axes" framing in the README).

## 2. ProbeGen: 4-stage constraint-driven probe synthesis

Generating probe pairs by hand does not scale, and generating them with a
single unconstrained LLM call produces prompts that only loosely honor the
intended logical relationship. ProbeGen (`src/consistencybench/probegen.py`)
splits generation into four stages so that the constraint is enforced
structurally rather than hoped for:

1. **Constraint Specification** (`build_constraint_spec`) — pure Python, no
   LLM call. Looks up the family's formal constraint, consistency rule, and
   probe structure from `theory.FAMILY_FORMAL_SPEC`, and attaches the target
   semantic-distance band for the requested difficulty level.
2. **Semantic Instantiation** (`build_probegen_prompt` + one `call_model`
   call) — a single generator-model call (`GENERATOR_MODEL`, Gemini 2.5 Pro
   in the reference run) that instantiates the abstract constraint over a
   concrete domain (general / science / ethics) at a target difficulty. The
   generator model is given the *exact* delta formula and told what each
   difficulty band means, not just an adjective like "hard".
3. **Difficulty Calibration** — folded into the same call as Stage 2: the
   prompt explicitly asks the generator to maximize surface divergence
   between `prompt_a` and `prompt_b` while perfectly preserving the logical
   link, for the "hard" band. This is checked after the fact (Stage 4 below
   is about logical validity; the actual `delta` is recomputed independently
   in `theory.compute_semantic_distance` once probes come back, and probes
   are labeled with their **measured** difficulty, not just their requested
   one — see `probegen.generate_full_dataset`).
4. **Constraint Verification** (`verify_probe`) — a second LLM call (with a
   different, verification-specific prompt) checks every generated probe
   against five criteria: logical validity, semantic preservation under the
   transformation, type correctness of the scoring hint, scoring
   compatibility (rbs probes must have clearly extractable yes/no answers),
   and standalone-ness (no cross-references between prompt_a and prompt_b).
   To control cost, only a ~10% spot-check sample per batch is verified;
   failures are dropped and the batch is topped up. **Rejection rate in the
   reference run: 4.2%.**

Probes are deduplicated within and across batches using trigram-Jaccard
similarity on `prompt_a` (`probegen.deduplicate_probes`, threshold 0.82/0.88)
to prevent near-duplicate probes from inflating apparent sample size.

## 3. The difficulty metric: δ (semantic distance)

Difficulty is not a label chosen by the generator model — it is *measured*,
using a real sentence-embedding model (`all-MiniLM-L6-v2`), after generation:

```
delta(pA, pB) = alpha * d_lex + beta * d_sem + gamma * d_syn      (alpha=beta=0.4, gamma=0.2)

d_lex = 1 - Jaccard(tokens(pA), tokens(pB))                       lexical overlap
d_sem = (1 - cosine_similarity(embed(pA), embed(pB))) / 2         embedding distance, in [0,1]
d_syn = 1[syntactic-depth-bucket(pA) != syntactic-depth-bucket(pB)]  length-bucket mismatch
```

`delta in [0, 1]`; higher means the two prompts are more surface-divergent
while (by construction, verified in Stage 4) still logically linked.
Difficulty bands: **easy** `delta < 0.35`, **medium** `0.35 <= delta < 0.65`,
**hard** `delta >= 0.65`. This makes "hard" an auditable, reproducible
property of the probe text, not an opaque generator-model judgment call —
anyone can recompute `delta` for any probe pair independently
(`theory.compute_semantic_distance`).

## 4. The evaluation harness

`src/consistencybench/harness.py` defines a single abstract interface,
`ModelBackend`, with two implementations:

- **`APIBackend`** — wraps every frontier/production model via the
  OpenRouter API (`client.call_model`), with retry/backoff and cost tracking.
- **`LocalHFBackend`** — loads an open-weight model locally via
  `transformers`, with **greedy decoding** to mirror `temperature=0` on the
  API side, and an additional `query_with_activations` method that returns
  the full residual-stream hidden state (every layer, final prompt token)
  alongside the generated text — the entry point for Part B.

Both backends are driven by the same `EvaluationHarness`, so no code path
differs between "the 16 API models" and "the interpretability testbed model"
except which backend is plugged in. Every harness run computes a
**`run_id`** — a SHA-256 hash of `{backend_type, model_id, system_prompt,
seed}` — and checkpoints under that hash, so:

- Two runs with different system prompts, models, or seeds never silently
  merge results in the same checkpoint file.
- An interrupted run resumes exactly where it left off (`resume=True`,
  checked at the `(probe_id, model)` level).

## 5. Scoring: RBS + LJS ensemble

Two independent scoring methods are used, routed by transformation family
(`config.FAMILY_SCORING`):

- **Rule-Based Scoring (RBS)** — for **Reversal**, **Ordering**, and
  **Complement**, both expected answers are simple yes/no. `extract_yn`
  parses the first sentence of each response against a small regex set; if
  both parse, consistency is a deterministic function of the extracted
  yes/no pair and the family's rule (`same` or `opposite`,
  `config.RBS_RULES`). Zero LLM calls, zero judge bias.
- **LLM-as-Judge (LJS)** — for **Composition** and **Equivalence**, the
  logical relationship between two free-form answers can't be reduced to a
  yes/no comparison. A judge model (`GENERATOR_MODEL` by default) is shown
  both prompts, both responses, and the intended logical constraint, and
  asked only to judge logical consistency — explicitly instructed to ignore
  factual accuracy or answer quality (`scoring.LJS_PROMPT`).
- **Fallback**: if RBS is inconclusive (an answer didn't parse as yes/no)
  *and* a judge is available, LJS is used as a backup even for RBS-eligible
  families (`scoring.score_result`).

### Validity checks on the scoring pipeline itself

- **Cross-judge validation** (`scoring.cross_judge_validation`) — a sample of
  LJS-scored results are re-scored with a *different* judge model. Agreement,
  Cohen's kappa, and Krippendorff's alpha between the two judges quantify
  whether LJS verdicts are an artifact of one particular judge's biases.
- **Inter-annotator agreement** (`agreement.compute_iaa`) — the inverse
  check: a sample of *RBS*-scored results (which have a mechanically
  determined ground truth) are also scored by an LLM judge. High agreement
  here is evidence that LJS — which has no such ground truth to check itself
  against on Composition/Equivalence — is trustworthy on the subset where it
  *can* be checked.

## 6. Consistency Profiles (CP)

A single Inconsistency Rate (IR) number collapses a 5-dimensional failure
structure into one scalar. `src/consistencybench/profiles.py` computes, for
every model, a 5D vector of per-family IR:

```
CP(model) = [ IR(model, composition), IR(model, reversal), IR(model, complement),
              IR(model, ordering),    IR(model, equivalence) ]
```

Two models can have near-identical overall IR while failing on completely
different families — CP makes that visible (see Figure 4, the radar plots).
`profiles.profile_distance_matrix` computes pairwise Euclidean distance
between CP vectors (which models "fail the same way"), and
`profiles.family_rank_correlation` computes the Spearman correlation of
per-family IR *across* models — a high correlation is evidence that family
difficulty is a property of the *task* (some logical transformations are
inherently harder for current LLMs), not an idiosyncrasy of any one model's
training.

## 7. Intervention: Baseline → CR → SC → FTSC

Run on a sample of **hard** probes (highest measured delta) across four
representative models (`intervention.INTERVENTION_MODELS`), four
conditions are compared (`intervention.run_intervention_conditions`):

| Condition | What changes |
|---|---|
| **Baseline** | Independent queries, no consistency instruction |
| **CR** (Consistency Reminder) | System prompt generically asks the model to be logically consistent across related questions |
| **SC** (Self-Check) | `prompt_b` is prefixed with the model's own `prompt_a` answer and asked to be consistent with it |
| **FTSC** (Family-Targeted Self-Check) | SC, plus the system message *names the specific transformation family and its formal consistency rule* |

FTSC isolates a specific question: does telling a model *which kind* of
logical constraint applies help more than a generic "be consistent"
instruction? The comparison is meaningful only because SC and FTSC share
almost the same prompt structure — FTSC's only addition is the
family-specific rule text (`intervention.get_ftsc_prompt`).

## 8. Consistency-Calibration Score (CCS)

CCS (`src/consistencybench/calibration.py`) is Expected Calibration Error
(ECE), adapted to a consistency framing: does a model's *expressed
confidence* predict whether it is *actually* about to be inconsistent?

- A predicted probability of inconsistency, `p_inconsistent`, is derived from
  the score method: for LJS results, from the judge's own stated confidence
  (`ljs_confidence`, mapped `high/medium/low -> 0.90/0.65/0.40`, then flipped
  if the verdict was "consistent"); for RBS results, from a lightweight
  hedge-language heuristic on the response text itself
  (`calibration.rbs_confidence`).
- CCS bins `p_inconsistent` into deciles and computes the weighted mean
  absolute gap between predicted and observed inconsistency rate per bin —
  `calibration.compute_ccs`, the same construction as classic ECE.

A model can have high IR but *low* CCS (well-calibrated: its own
low-confidence answers are disproportionately the ones that fail) — a
meaningfully safer deployment profile than the same IR with no such signal,
since low-confidence-flagged answers can be routed to a human or a stronger
model. Figure 8 plots CCS against IR to show these are largely orthogonal.

## 9. Hint sensitivity

`src/consistencybench/hint_sensitivity.py` asks a related but distinct
question from everything else in the pipeline: is a model's answer to a
*single* prompt stable under **purely social/epistemic pressure** — a vague,
unsupported appeal to authority or consensus, with zero new evidence
attached? Starting from pairs the model was *already consistent on*, a hint
template is appended only to `prompt_b`, and two rates are measured:

- **Hint Flip Rate** — how often the yes/no orientation of the answer to
  `prompt_b` changes once hinted.
- **Hint-Induced Inconsistency** — of pairs consistent at baseline, how many
  become logically inconsistent (relative to the *unhinted* `response_a`)
  once `prompt_b` is hinted.

This sits closer to sycophancy than to logical inconsistency, but is
measured with the same infrastructure (`scoring.score_result`,
`scoring.extract_yn`) and reported alongside IR/CCS as a third axis of answer
instability in the leaderboard.

## 10. Scaling analysis

`src/consistencybench/scaling.py` answers a practical question for anyone
adapting ConsistencyBench to a *new* model on a limited API budget: how many
probes are actually needed before the IR estimate stabilizes? For each
model, bootstrapped sub-samples at increasing size are drawn (30 resamples
per size), and the smallest sample size at which the bootstrapped mean IR
converges to within ±1.5 percentage points of the full-dataset IR is
reported — the recommended minimum evaluation budget.

## 11. Statistical tests

`src/consistencybench/statistics.py` runs the full battery
(`run_statistical_tests`):

1. Per-model IR with 95% CI (normal approximation via `scipy.stats.sem`)
2. Chi-squared test of independence over the (model × family) inconsistency
   contingency table — does the *pattern* of failures differ by model?
3. Kruskal-Wallis test over difficulty bands — does IR increase with delta?
4. Mann-Whitney U test, ethics domain vs. all other domains (one-sided) +
   Cohen's d effect size
5. Kendall's tau and Spearman's rho between MMLU accuracy and IR — is
   inconsistency just "being a weaker model" in disguise?
6. Kendall's tau between parameter count and IR — does scale predict
   consistency?
7. Paired t-tests over four **ablation pairs**
   (`config.ABLATION_PAIRS`) that isolate one training axis at a time (e.g.
   reasoning-trained vs. not, same base model family) while holding org and
   scale roughly fixed

and renders eight publication-ready LaTeX tables
(`statistics.generate_latex_tables`) covering main results, difficulty,
domain, Consistency Profiles, intervention, the statistical-test summary,
hint sensitivity, and the interpretability extension.

## 12. Reproducibility guarantees

- **Determinism**: `TEMPERATURE = 0.0` for every evaluated model;
  `LocalHFBackend` uses greedy decoding (`do_sample=False`) to match.
- **Global seeding**: `harness.set_global_seed(42)` seeds Python's `random`,
  NumPy, and PyTorch (+ CUDA) at the start of every harness run.
- **Config-hashed run IDs**: any change to backend, model, system prompt, or
  seed produces a new `run_id` and therefore a new checkpoint namespace —
  results from different configurations can never silently mix.
- **Resumability everywhere**: every long-running stage (`probegen`,
  `harness.run`, `scoring.score_all`, activation extraction) checkpoints
  incrementally and picks up from where it left off on restart.
- **Independent auditability of difficulty**: `delta` is recomputed from the
  probe text itself, not trusted from the generator model's self-report.
