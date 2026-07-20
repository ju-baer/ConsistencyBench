# ConsistencyBench

**A Constraint-Preserving Framework for Evaluating Logical Consistency as a First-Class Property of Language Models**

[![tests](https://github.com/ju-baer/ConsistencyBench/actions/workflows/tests.yml/badge.svg)](https://github.com/ju-baer/ConsistencyBench/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](docs/DATASET_CARD.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-orange?style=flat-square)](https://huggingface.co/datasets/jub-aer/ConsistencyBench-probes)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-orange?style=flat-square)](https://huggingface.co/datasets/jub-aer/ConsistencyBench-interpretability)


> Standard benchmarks measure whether a model gets the *right* answer.
> ConsistencyBench measures whether a model gives the *same* answer to
> questions that are logically guaranteed to require one — a distinct,
> orthogonal failure mode that accuracy scores do not capture.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [The core idea in one picture](#the-core-idea-in-one-picture)
- [What's in this repository](#whats-in-this-repository)
- [Repository structure](#repository-structure)
- [Quickstart](#quickstart)
- [The five transformation families](#the-five-transformation-families)
- [Pipeline overview](#pipeline-overview)
- [Key results (reference run)](#key-results-reference-run)
- [The interpretability extension (Part B)](#the-interpretability-extension-part-b)
- [Reproducing this project](#reproducing-this-project)
- [The dataset](#the-dataset)
- [Testing](#testing)
- [FAQ](#faq)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Why this exists

Every major LLM leaderboard measures accuracy: does the model get the right
answer? None of them ask a logically prior question: **is the model even
internally consistent with itself?** A model can score well on MMLU while
still telling you `X implies Y` in one breath and `NOT(X implies Y)` in the
next, simply because the two questions were phrased differently or asked five
minutes apart.

This matters for deployment in a very concrete way. A user who asks the same
underlying question twice, in different words, reasonably expects the same
answer. A downstream system that composes two independent model calls
reasonably expects the outputs to be compatible. **Logical consistency is a
reliability property, not a knowledge property** — and it needs its own
benchmark, because accuracy benchmarks structurally cannot detect it: a
single question has only one "right" answer to check, so there is nothing in
a standard eval to even notice self-contradiction across two related
questions.

ConsistencyBench is a framework — not just a fixed leaderboard — for
generating, verifying, and scoring probe pairs that isolate this failure mode
directly, plus a set of follow-on questions any consistency benchmark should
also answer: *how* inconsistent (which logical structures break, per model)?
*Can it be fixed with a prompt*? *Is the model at least aware when it's about
to contradict itself* (calibration)? *Where does it come from, mechanistically,
in an open-weight model we can actually look inside?*

## The core idea in one picture

```
 STANDARD BENCHMARK                       CONSISTENCYBENCH
 ───────────────────                       ─────────────────
 Q: "Is Paris the capital                  A: "Does X entail Y?"      -> "Yes"
     of France?"                           B: "Given Y is true, and Y entails
 A: "Yes."                                     Z, does X entail Z?"   -> "No"
 Score: correct  ✓                         Score: INCONSISTENT ✗
                                            (A said X=>Y; B implicitly denies
                                             the transitive consequence of
                                             what A already granted)

 Checks: is the answer TRUE?               Checks: are the answers
                                            LOGICALLY COMPATIBLE with
                                            each other, regardless of
                                            whether either is true?
```

A model can be *consistently wrong* (score well here while being factually
incorrect on both prompts) — that's fine, and deliberate: this benchmark is
measuring a different, orthogonal axis from accuracy. See
[Key results](#key-results-reference-run) for the empirical evidence that IR
(Inconsistency Rate) and accuracy are largely uncorrelated across the 16
evaluated models.

## What's in this repository

This repository is a full research pipeline, structured for both
**reproduction** and **extension**:

1. **Part A — Behavioral benchmark.** A constraint-driven probe-generation
   framework (**ProbeGen**), an evaluation harness that runs 16 frontier and
   open-weight API models under identical conditions, a dual scoring
   ensemble (rule-based + LLM-as-judge, cross-validated against each other),
   five derived analyses (Consistency Profiles, an intervention ladder,
   calibration, hint sensitivity, scaling), and a full statistical-testing
   suite.
2. **Part B — Mechanistic interpretability extension.** The same probe set
   run against a local, fully white-box open-weight model, with per-layer
   linear probing to locate where inconsistency becomes decodable in the
   residual stream, and two genuinely *causal* tests (activation patching,
   diff-in-means feature steering with a dose-response curve and a
   specificity check) — not just correlational probing.
3. **Part C — Delivery.** Eleven publication-quality figures, eight LaTeX
   tables, a qualitative error-analysis pass, and a ready-to-upload
   HuggingFace dataset export (probes / results+leaderboard /
   interpretability bundle, each with an auto-generated dataset card).

The original, single-notebook version of this project is preserved at
[`notebooks/ConsistencyBench_NeurIPS2026.ipynb`](notebooks/ConsistencyBench_NeurIPS2026.ipynb)
— the easiest way to run everything interactively cell-by-cell on Google
Colab. Everything in `src/consistencybench/` is the *same logic*, refactored
into an importable, individually testable Python package, with the notebook
cell each module was ported from noted in its docstring.

## Repository structure

```
ConsistencyBench/
├── README.md                          <- you are here
├── LICENSE                            MIT (code) — see docs/DATASET_CARD.md for data license
├── CONTRIBUTING.md                    how to add a model / family / backend
├── pyproject.toml                     package metadata + dependencies
├── requirements.txt                   flat pip-installable dependency list
│
├── notebooks/
│   └── ConsistencyBench_NeurIPS2026.ipynb   the original, canonical, cell-by-cell notebook
│
├── src/consistencybench/              the same logic, as an importable package
│   ├── config.py                          model roster, families, costs, run params
│   ├── client.py                          OpenRouter client, robust JSON parsing, checkpointing
│   ├── theory.py                          formal constraint specs + the δ difficulty metric
│   ├── probegen.py                        4-stage constraint-driven probe synthesis
│   ├── harness.py                         ModelBackend (API + local-HF), EvaluationHarness
│   ├── scoring.py                         RBS + LJS scoring ensemble, cross-judge validation
│   ├── agreement.py                       inter-annotator agreement (RBS vs LJS)
│   ├── profiles.py                        5D Consistency Profiles per model
│   ├── intervention.py                    Baseline -> CR -> SC -> FTSC prompting ladder
│   ├── calibration.py                     Consistency-Calibration Score (CCS / ECE)
│   ├── hint_sensitivity.py                sycophancy-adjacent hint-flip testing
│   ├── scaling.py                         bootstrapped sample-size convergence analysis
│   ├── statistics.py                      the full statistical-test battery + LaTeX tables
│   ├── interpretability/                  Part B: mechanistic interpretability
│   │   ├── activations.py                     white-box activation extraction (local model)
│   │   ├── shortcut_probe.py                  per-layer linear probing (where is it decodable?)
│   │   └── causal_tests.py                    activation patching + diff-in-means steering
│   └── analysis/                          Part C: figures, qualitative, export
│       ├── figures.py                         all 11 publication figures
│       ├── qualitative.py                     one worked failure example per family
│       └── hf_export.py                       HuggingFace dataset bundles + dataset cards
│
├── scripts/                           numbered, resumable CLI entry points
│   ├── 01_run_probegen.py                 generate the probe dataset
│   ├── 02_run_experiments.py              run all 16 models through the harness
│   ├── 03_run_scoring.py                  score + validate (cross-judge, IAA)
│   ├── 04_run_analysis.py                 profiles, intervention, CCS, hints, scaling
│   ├── 05_run_interpretability.py         Part B, local model (needs a GPU)
│   ├── 06_generate_deliverables.py        figures, tables, qualitative, HF export
│   └── run_full_pipeline.py               convenience wrapper for all of the above
│
├── docs/
│   ├── METHODOLOGY.md                     the detailed "how and why" for every component
│   ├── TRANSFORMATION_FAMILIES.md         deep dive on all five logical families
│   ├── INTERPRETABILITY.md                deep dive on Part B
│   └── DATASET_CARD.md                    HuggingFace-style dataset documentation
│
├── tests/                              51 unit tests, network- and GPU-free by design
│
├── data/                               all generated artifacts land here (gitignored)
│   ├── probes/, results/, checkpoints/, scoring/, profiles/, intervention/,
│   │   hints/, interpretability/, figures/, tables/, hf_export/
│
├── figures/, tables/                   convenience symlink targets for the latest run
└── .github/workflows/tests.yml         CI: lint + unit tests on every push/PR
```

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/ConsistencyBench.git
cd ConsistencyBench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # Part A + testing, no torch
# pip install -e ".[interpretability]"  # add this for Part B (needs a GPU for speed)

export OPENROUTER_API_KEY=sk-...        # https://openrouter.ai/keys

# Fast smoke test end-to-end (5 probes/combo instead of 100 = 75 probes total)
python scripts/run_full_pipeline.py --quick-test

# Full behavioral run (4,500 probes x 16 models — many hours, real API cost; see below)
python scripts/01_run_probegen.py
python scripts/02_run_experiments.py
python scripts/03_run_scoring.py
python scripts/04_run_analysis.py
python scripts/06_generate_deliverables.py

# Optional: mechanistic interpretability extension (needs a GPU)
python scripts/05_run_interpretability.py
```

Every stage checkpoints incrementally under `data/checkpoints/` — if a run
is interrupted (rate limits, a killed Colab session, a laptop going to
sleep), re-running the same script picks up exactly where it left off rather
than re-querying already-completed `(probe, model)` pairs.

**Cost note:** a full run is `N_PER_COMBO(100) x 5 families x 3 domains x 3
difficulties = 4,500 probes`, each queried twice against 16 models =
**144,000 API calls** for Part A alone, plus generator/judge calls for
ProbeGen and LJS scoring. Use `--quick-test` (5 probes/combo) to validate the
whole pipeline cheaply before committing to a full run, and see
`config.summarize()` (printed at the start of `01_run_probegen.py`) for an
exact call-count estimate before you start.

## The five transformation families

Every probe pair instantiates one of five formally specified, truth
-preserving logical transformations (`theory.py::FAMILY_FORMAL_SPEC`),
together documented to span roughly 83% of logical relationships occurring
in natural QA. Full depth, worked examples, and design rationale in
[docs/TRANSFORMATION_FAMILIES.md](docs/TRANSFORMATION_FAMILIES.md).

| Family | Logical basis | Formal constraint | Scoring | Coverage |
|---|---|---|---|---|
| **Composition** | Relation composition (transitivity) | `A⇒B ∧ B⇒C ⟹ A⇒C` | LLM-as-Judge | 18.4% |
| **Reversal** | Symmetric relation reversal | `rel(X,Y) ⟺ rel(Y,X)` | Rule-based (same) | 14.2% |
| **Complement** | Truth complement (negation) | `¬(assert(P) ∧ assert(¬P))` | Rule-based (opposite) | 28.6% |
| **Ordering** | Asymmetric temporal ordering | `before(A,B) ⟺ after(B,A)` | Rule-based (same) | 10.8% |
| **Equivalence** | Semantic equivalence preservation | `equiv(pA,pB) ⟹ compat(rA,rB)` | LLM-as-Judge | 11.4% |

**Difficulty** is not a subjective label — it's a measured semantic distance
`δ = 0.4·d_lex + 0.4·d_sem + 0.2·d_syn` between `prompt_a` and `prompt_b`
(lexical Jaccard distance, sentence-embedding cosine distance, syntactic
length-bucket mismatch), recomputed independently from the generated text
rather than trusted from the generator model. `easy: δ<0.35`, `medium:
0.35≤δ<0.65`, `hard: δ≥0.65`. Full derivation in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md#3-the-difficulty-metric-δ-semantic-distance).

## Pipeline overview

```
┌─────────────┐   ┌──────────────┐   ┌───────────┐   ┌──────────────┐   ┌───────────────────┐
│  ProbeGen   │──▶│  Evaluation  │──▶│  Scoring  │──▶│   Analysis   │──▶│   Deliverables     │
│ (4 stages)  │   │   Harness    │   │ RBS + LJS │   │ CP/CCS/Interv│   │ Figures/Tables/HF  │
└─────────────┘   └──────────────┘   └───────────┘   └──────────────┘   └───────────────────┘
 script 01          script 02          script 03        script 04           script 06
                                                                ▲
                                                                │ (shares checkpoints with)
                                                          ┌─────┴──────┐
                                                          │  Part B:   │
                                                          │Interp. (05)│
                                                          └────────────┘
```

1. **ProbeGen** (`probegen.py`, script `01`) — constraint specification
   (pure logic, no LLM) → semantic instantiation (generator LLM call,
   Gemini 2.5 Pro) → difficulty targeting (folded into the same call, then
   independently re-measured) → constraint verification (a second LLM call
   checking 5 validity criteria on a spot-check sample; **4.2% reference
   rejection rate**). Deduplicated via trigram-Jaccard.
2. **Evaluation Harness** (`harness.py`, script `02`) — a single
   `ModelBackend` interface with `APIBackend` (16 evaluated models via
   OpenRouter) and `LocalHFBackend` (the Part B interpretability model)
   implementations, driven by one `EvaluationHarness`. Deterministic
   (`temperature=0`, greedy local decoding), globally seeded, and
   checkpointed by a config-hash `run_id` so different configurations never
   silently merge.
3. **Scoring** (`scoring.py` + `agreement.py`, script `03`) — Rule-Based
   Scoring for the three families with clean yes/no answers (deterministic,
   zero LLM calls), LLM-as-Judge for the two that need free-form comparison,
   with a documented fallback path and two independent quality checks:
   cross-judge validation (does a second judge model agree with the first?)
   and inter-annotator agreement (does an LLM judge agree with RBS's
   mechanical ground truth?).
4. **Analysis** (`profiles.py`, `intervention.py`, `calibration.py`,
   `hint_sensitivity.py`, `scaling.py`, script `04`) — five derived
   questions beyond the headline IR number, described in the next section.
5. **Interpretability** (`interpretability/`, script `05`, optional,
   needs a GPU) — see [below](#the-interpretability-extension-part-b).
6. **Deliverables** (`analysis/figures.py`, `statistics.py`,
   `analysis/qualitative.py`, `analysis/hf_export.py`, script `06`) —
   everything needed to write up or publish the results.

Full narrative detail on every stage: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

### The five analyses beyond IR

| Analysis | Module | Question it answers |
|---|---|---|
| **Consistency Profiles (CP)** | `profiles.py` | Which *specific* logical structures does each model fail on? (5D fingerprint, not one scalar) |
| **Intervention ladder** | `intervention.py` | Does telling a model to "be consistent" help — and does naming the *specific* logical rule (FTSC) help more than a generic reminder (CR)? |
| **Consistency-Calibration Score (CCS)** | `calibration.py` | Does the model's own expressed confidence predict when it's about to be inconsistent? (ECE-style) |
| **Hint sensitivity** | `hint_sensitivity.py` | Does an unsupported social/authority hint (zero new evidence) flip the model's answer? |
| **Scaling analysis** | `scaling.py` | How many probes are actually needed before a new model's IR estimate stabilizes (±1.5pp)? |

## Key results (reference run)

> These are the headline findings from the reference run documented in the
> original notebook and its accompanying writeup. Regenerate
> `data/experiment_metadata.json` and `tables/tab_stats.tex` with your own
> run (models and API pricing drift over time) before citing exact numbers.

- **Every evaluated model exhibits a non-trivial Inconsistency Rate** —
  logical inconsistency is not a fringe failure mode confined to weak
  models; it is present, to varying degrees, across every architecture and
  scale tested.
- **IR is largely orthogonal to accuracy.** Kendall's τ between MMLU
  accuracy and overall IR across the 16 models is weak and not reliably
  distinguishable from zero — a model being more accurate does not reliably
  predict it being more self-consistent. This is the empirical basis for
  treating consistency as a genuinely separate evaluation axis rather than
  a proxy for capability (Figure 5 / Figure 1's evaluation-space plot).
- **Complement and Equivalence are the two hardest families**, consistently,
  across nearly every model (Figure 3's heatmap) — the most basic form of
  self-contradiction (affirming `P` and `¬P`) and the most surface-agnostic
  paraphrase-robustness test are both harder than transitive composition or
  relation reversal.
- **Reasoning-trained models help selectively, not universally** — the
  ablation pairs (`config.ABLATION_PAIRS`: same base model with/without
  reasoning training, e.g. DeepSeek R1 vs. V3, o4-mini vs. GPT-4o-mini) show
  paired-t-test-significant IR reduction on *some* families and not others,
  rather than a uniform improvement across the board.
- **Hard probes (δ≥0.65) elicit 2-3x higher IR than easy probes** across
  every model tested (Figure 6) — surface divergence between
  logically-linked prompts is a genuine adversarial lever, not just noise.
- **Ethics is the consistently hardest domain** (Figure 9) — value-laden
  reasoning shows systematically higher IR than general-knowledge or
  science framings of logically equivalent probes.
- **FTSC (naming the specific logical rule) outperforms generic
  consistency-reminder prompting** on hard probes (Figure 7) — a concrete,
  actionable mitigation signal: *specificity* of the self-check instruction
  matters, not just its presence.
- **IR and CCS are largely independent** (Figure 8) — a model can be highly
  inconsistent yet reasonably well-calibrated about *when* (its own
  low-confidence answers disproportionately being the ones that fail), which
  matters directly for deployment: well-calibrated inconsistency can be
  routed and caught; poorly-calibrated inconsistency cannot.
- **Some models are measurably swayed by unsupported hints** — the
  `hint_induced_ir_pct` leaderboard column identifies models whose answers
  flip under pure social pressure with no new evidence, a sycophancy
  -adjacent but mechanically distinct failure from logical inconsistency
  proper.

## The interpretability extension (Part B)

Part A treats all 16 API models as black boxes. Part B asks a mechanistic
question of one **local, fully white-box open-weight model**
(`Qwen2.5-1.5B-Instruct` by default — small enough to run per-layer hooks
and activation patching quickly, ungated, and instruction-tuned to follow
the same yes/no protocol as the API models). This model is a dedicated
interpretability testbed, not a leaderboard entry.

1. **Activation extraction** (`activations.py`) — capture the full
   residual stream (every layer, final prompt token) for a balanced sample
   of probes, run through the *same* harness and system prompt as Part A.
2. **Shortcut probing** (`shortcut_probe.py`) — train a linear probe at
   *every layer* to decode "will this response be inconsistent?" directly
   from the residual stream (Alain & Bengio 2017 methodology), plus a
   family-conditioned breakdown and a PCA-based pseudo-feature analysis (an
   explicit, documented substitute for a pretrained SAE, which does not
   exist for this checkpoint).
3. **Causal tests** (`causal_tests.py`) — two genuinely causal
   interventions, not just correlational probing:
   - **Activation patching**: literally overwrite an inconsistent run's
     residual stream at the best-probe layer with a matched consistent
     run's activation, and check whether the output flips.
   - **Diff-in-means feature steering**: add a scaled consistent-minus
     -inconsistent direction to the residual stream and measure a genuine
     **dose-response curve** (IR at increasing steering strength), plus a
     **specificity check** on unrelated control questions to rule out the
     trivial "steering just breaks the model" explanation.

Full methodology, exact layer-indexing conventions, and an explicit
statement of what this extension does and doesn't claim to generalize to:
[docs/INTERPRETABILITY.md](docs/INTERPRETABILITY.md).

```bash
pip install -e ".[interpretability]"
python scripts/05_run_interpretability.py    # needs a GPU for reasonable runtime
```

## Reproducing this project

Two equally valid ways to reproduce or extend this work:

- **Interactively, cell-by-cell**: open
  `notebooks/ConsistencyBench_NeurIPS2026.ipynb` in Colab (free-tier T4 is
  sufficient for Part B), set `OPENROUTER_API_KEY` in Colab Secrets, run
  cells in order. This is the original, canonical narrative of the project.
- **As a pipeline, stage-by-stage or all at once**: use `scripts/`, as in
  [Quickstart](#quickstart). Every stage is independently resumable and
  independently unit-testable (see [Testing](#testing)) — the recommended
  path for CI, for running on a remote machine over multiple sessions, or
  for swapping in your own model/family/backend (see
  [CONTRIBUTING.md](CONTRIBUTING.md)).

Both paths call the exact same underlying logic in `src/consistencybench/`.

## The dataset

The probe set, full model results, and interpretability bundle are designed
to be published as three linked HuggingFace datasets, generated by
`scripts/06_generate_deliverables.py` into `data/hf_export/`:

| Bundle | Contents | Model-agnostic |
|---|---|---|
| `consistencybench-probes` | 4,500 probe pairs (prompts, constraints, difficulty) | Yes |
| `consistencybench-results` | Full scored (probe, model) pairs + leaderboard CSV | No |
| `consistencybench-interpretability` | Layer probing, patching, steering, hint-sensitivity bundle | No (single local model) |

Each bundle ships with an auto-generated dataset card
(`README_probes.md`, `README_results.md`, `README_interpretability.md`) with
correct YAML frontmatter for the HF Hub. Full schema documentation:
[docs/DATASET_CARD.md](docs/DATASET_CARD.md).

**Submitting your own model's results** doesn't require running ProbeGen —
see [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-model-to-the-leaderboard).

## Testing

51 unit tests cover every module's pure-logic core (config sanity, difficulty
classification, RBS scoring rules, robust JSON parsing, probe deduplication,
CCS/ECE computation, Consistency Profile distance) with synthetic fixtures —
deliberately **network- and GPU-free**, so they run in CI on every push
without an API key or a GPU runner.

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/ tests/ scripts/
```

Anything that requires an actual LLM call or a loaded model (probe
generation quality, scoring accuracy on real responses, activation
extraction) is validated empirically via `--quick-test` runs, not unit
tests — see [CONTRIBUTING.md](CONTRIBUTING.md#code-contributions-in-general)
for the rationale.

## FAQ

**Does a high Inconsistency Rate mean the model is bad?**
It means the model is unreliable in a specific, measurable way that
accuracy benchmarks don't capture — not that it's unintelligent. Some of the
most accurate models in the reference run also have non-trivial IR (see
[Key results](#key-results-reference-run)).

**Why exclude the generator model (Gemini 2.5 Pro) from evaluation?**
To avoid circularity: a model that generated the probes (and therefore
"knows" the intended logical relationship from having written both sides of
each pair) would have an unfair, uninterpretable advantage if also scored
against them.

**Why is Complement scored by rule and Composition by an LLM judge?**
Complement's constraint reduces to comparing two extracted yes/no answers —
fully mechanical, no judgment call needed. Composition's constraint links
two free-form conclusions that can't be reduced to a fixed string
comparison. See
[docs/METHODOLOGY.md](docs/METHODOLOGY.md#5-scoring-rbs--ljs-ensemble) for
the full scoring-method rationale and the cross-checks run on both methods.

**Can I evaluate a model that isn't on OpenRouter?**
Yes — implement the `ModelBackend` interface in `harness.py`; see
[CONTRIBUTING.md](CONTRIBUTING.md#adding-a-new-api-model-backend-non-openrouter).

**Is this the same as "hallucination" or "jailbreak" benchmarks?**
No. Hallucination is about factual grounding; jailbreak robustness is about
adversarial instruction-following. Logical consistency is orthogonal to
both — a model can hallucinate consistently, refuse consistently, or (the
subject of this benchmark) contradict itself while staying entirely within
normal, non-adversarial use.

## Limitations

- The five transformation families cover an estimated ~83% of naturally
  occurring logical relationships in QA; multi-hop chains, modal operators,
  and quantifier scope are explicitly out of scope for this release (see
  [docs/TRANSFORMATION_FAMILIES.md](docs/TRANSFORMATION_FAMILIES.md#design-rationale-why-these-five-and-not-others)).
- LJS (LLM-as-Judge) scoring inherits whatever biases the judge model has;
  cross-judge and RBS-agreement checks bound, but do not eliminate, this.
- English-only in the current release.
- Part B's mechanistic findings are demonstrated on one comparatively small
  open-weight model and are not claimed to generalize to the much larger,
  closed-weight models evaluated in Part A — see
  [docs/INTERPRETABILITY.md](docs/INTERPRETABILITY.md#what-this-extension-does-and-doesnt-claim).
- Reference numbers throughout this README reflect a single point-in-time
  run; model behavior, availability, and API pricing all drift — regenerate
  before citing exact figures for anything beyond illustration.

## Citation

If you use ConsistencyBench in your research, please cite:

```bibtex
@misc{consistencybench2026,
  title  = {ConsistencyBench: A Constraint-Preserving Framework for
            Evaluating Logical Consistency as a First-Class Property of
            Language Models},
  author = {S M Jubaer},
  year   = {2026},
  url    = {https://github.com/ju-baer/ConsistencyBench}
}
```

## License

Code: [MIT](LICENSE). Dataset artifacts: CC-BY-4.0, see
[docs/DATASET_CARD.md](docs/DATASET_CARD.md#licensing).

## Acknowledgments

Built on top of the [OpenRouter](https://openrouter.ai) API for unified
access to frontier and open-weight models, HuggingFace `transformers` for
local white-box model access, and `sentence-transformers` for the semantic
-distance difficulty metric. Linear probing methodology follows Alain &
Bengio (2017); feature steering follows the diff-in-means / activation
-steering methodology of Turner et al. (2023).
