"""
Part B — Mechanistic Interpretability Extension.

Everything in the behavioral benchmark (Part A) treats every model as a black
box: text in, text out. That characterizes *how often* inconsistency happens,
but never *why* — what's actually happening inside the network when it
produces two incompatible answers.

This subpackage switches to `LocalHFBackend` to load a small open-weight model
(Qwen2.5-1.5B-Instruct) with full white-box access, and runs three stages:

  activations     — capture residual-stream activations for every layer at the
                     final prompt token, for consistent and inconsistent runs
  shortcut_probe  — train per-layer linear probes to decode "will this
                     response be inconsistent?" directly from activations,
                     localizing *where* reasoning instability emerges
  causal_tests    — activation patching (does transplanting a consistent-run
                     activation fix an inconsistent run?) and diff-in-means
                     feature steering (does adding a steering vector reduce
                     inconsistency rate causally, with a dose-response curve
                     and a specificity check on unrelated questions?)

This local model is deliberately small and is **not** part of the 16-model API
leaderboard — it is a dedicated interpretability testbed, chosen for
GPU-friendliness and full activation access rather than for its ranking.
"""
