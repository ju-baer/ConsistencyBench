"""
ConsistencyBench — A Constraint-Preserving Framework for Evaluating Logical
Consistency as a First-Class Property of Language Models.

This package contains the full research pipeline described in the accompanying
NeurIPS 2026 Datasets & Benchmarks submission:

  Part A — Behavioral Benchmark (black-box, API models)
    config          : model roster, transformation families, cost tracking
    client          : OpenRouter client, robust JSON parsing, checkpointing
    theory          : formal transformation-family specs + difficulty metric (delta)
    probegen        : 4-stage constraint-driven probe synthesis framework
    harness         : ModelBackend interface (API + local HF) + EvaluationHarness
    scoring         : Rule-Based Scoring + LLM-as-Judge ensemble
    profiles        : 5D Consistency Profile (CP) vectors per model
    intervention    : Baseline -> CR -> SC -> FTSC mitigation ladder
    calibration     : Consistency-Calibration Score (CCS)
    hint_sensitivity: misleading-hint robustness probe (sycophancy-adjacent)

  Part B — Mechanistic Interpretability Extension (local open-weight model)
    interpretability.activations   : residual-stream activation extraction
    interpretability.shortcut_probe: per-layer linear probes ("where" it's decodable)
    interpretability.causal_tests  : activation patching + diff-in-means steering

Every module in this package is a directly importable, Colab-independent version
of the corresponding cell in notebooks/ConsistencyBench_NeurIPS2026.ipynb. The
scripts/ directory provides thin CLI wrappers that chain these modules into the
same pipeline the notebook runs interactively.
"""

__version__ = "1.0.0"
