# Contributing to ConsistencyBench

Thanks for considering a contribution. This project has two very different
kinds of contributors in mind — code contributors and dataset/leaderboard
contributors — so start with whichever section fits.

## Adding a model to the leaderboard

You don't need to touch the codebase to contribute results:

1. Fork/clone, `pip install -e .`, set `OPENROUTER_API_KEY` (or add your own
   `ModelBackend` — see below — if your model isn't on OpenRouter).
2. Evaluate at `temperature=0` with the exact system prompt in
   `src/consistencybench/harness.py::SYSTEM_PROMPT` (also duplicated at the
   top of `scripts/02_run_experiments.py`) on the released probe set
   (`consistencybench-probes` on HuggingFace, or `data/probes/` after
   running `scripts/01_run_probegen.py` yourself).
3. Score with `python scripts/03_run_scoring.py`.
4. Open a PR or issue with your `consistencybench_results.csv` /
   `consistencybench_leaderboard.csv` rows and a short description of the
   model (org, parameter count if known, release date).

## Adding a new transformation family

This is the highest-value code contribution. To keep a new family
consistent with the existing five:

1. Add a formal spec entry to `theory.py::FAMILY_FORMAL_SPEC` — logical
   basis, the formal constraint in symbolic notation, the consistency rule
   in plain English, and the probe structure.
2. Decide the scoring method: if both expected answers reduce to a clean
   yes/no, add a rule to `config.RBS_RULES` (`same`/`opposite`) and extend
   `scoring.rule_based_score`; otherwise use `ljs` and write a
   family-specific instruction block for `scoring.LJS_PROMPT` if the
   default framing doesn't fit.
3. Add the family to `config.TRANSFORMATION_FAMILIES` and
   `config.FAMILY_LABELS`.
4. Run `scripts/01_run_probegen.py --quick-test` and manually spot-check a
   handful of generated probes before a full run — Stage 4 verification
   catches gross errors but not every subtle one.
5. Add a `docs/TRANSFORMATION_FAMILIES.md` entry following the existing
   format (logical basis, formal constraint, consistency rule, probe
   structure, scoring, why it's hard).

## Adding a new API model backend (non-OpenRouter)

Implement the `ModelBackend` interface in `harness.py` (see `APIBackend` for
the reference shape: a `query(prompt, system) -> (response_text, error)`
method plus whatever cost-tracking makes sense for your provider) and wire
it into `scripts/02_run_experiments.py`.

## Code contributions in general

- Match the existing style: type-hinted function signatures, docstrings that
  explain *why* a design choice was made (not just what the code does), and
  a comment pointing back to the notebook cell a module was ported from,
  where applicable.
- Run the test suite (`pytest tests/`) and `ruff check src/ tests/` before
  opening a PR.
- Unit tests should not require network or GPU access — see `tests/` for the
  existing pattern of testing the pure-logic parts of each module (parsing,
  scoring rules, difficulty classification, deduplication) with synthetic
  fixtures, while leaving anything that calls an LLM or loads a model
  untested at the unit level.
- Large or generated files (checkpoints, activation dumps, figures, HF
  export bundles) should never be committed — see `.gitignore`.

## Reporting issues

Please include: which stage of the pipeline (`scripts/0N_*.py`), the model/
family/domain combination if relevant, and — for scoring disagreements —
the specific `probe_id` and both responses, since a scoring bug is usually
easiest to diagnose from a single concrete example.

## Code of conduct

Be respectful and constructive. This is a research benchmark; disagreements
about methodology are welcome and should be argued on the merits (with
data, where possible) rather than asserted.
