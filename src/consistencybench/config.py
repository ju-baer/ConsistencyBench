"""
Global configuration for ConsistencyBench: the evaluated model roster, the five
transformation families, ablation pairs, cost tracking, and the local
interpretability model used in Part B.

This is a direct, environment-independent port of Notebook Cell 2. The only
behavioral change from the notebook is that `BASE_DIR` now defaults to a local
`data/` directory instead of a Google Drive mount, and the API key is read from
the `OPENROUTER_API_KEY` environment variable instead of Colab Secrets.
"""
from __future__ import annotations

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Storage layout
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(os.environ.get("CONSISTENCYBENCH_DATA_DIR", REPO_ROOT / "data"))

SUBDIRS = [
    "probes", "results", "figures", "tables", "checkpoints",
    "hf_export", "intervention", "scoring", "profiles",
    "interpretability", "hints",
]


def ensure_dirs(base_dir: Path = BASE_DIR) -> Path:
    """Create the standard ConsistencyBench directory layout under base_dir."""
    for d in SUBDIRS:
        (base_dir / d).mkdir(parents=True, exist_ok=True)
    return base_dir


# ──────────────────────────────────────────────────────────────────────────────
# API key
# ──────────────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# ──────────────────────────────────────────────────────────────────────────────
# Generator: EXCLUDED from evaluation to prevent circularity bias
# ──────────────────────────────────────────────────────────────────────────────
GENERATOR_MODEL = "google/gemini-2.5-pro"

# ──────────────────────────────────────────────────────────────────────────────
# 16 evaluated models (API, black-box) across 6 paradigms
# ──────────────────────────────────────────────────────────────────────────────
MODELS = {
    "gpt4_1":        "openai/gpt-4.1",
    "gpt4o":         "openai/gpt-4o",
    "claude_opus":   "anthropic/claude-opus-4-5",
    "claude_sonnet": "anthropic/claude-sonnet-4-5",
    "grok3":         "x-ai/grok-3",
    "o4_mini":       "openai/o4-mini",
    "o3_mini":       "openai/o3-mini",
    "deepseek_r1":   "deepseek/deepseek-r1",
    "gemini_flash":  "google/gemini-2.0-flash-001",
    "gpt4o_mini":    "openai/gpt-4o-mini",
    "llama4":        "meta-llama/llama-4-maverick",
    "llama33_70b":   "meta-llama/llama-3.3-70b-instruct",
    "deepseek_v3":   "deepseek/deepseek-v3",
    "qwen3_235b":    "qwen/qwen3-235b-a22b",
    "mistral":       "mistralai/mistral-large-2411",
    "phi4":          "microsoft/phi-4",
}

MODEL_LABELS = {
    "gpt4_1": "GPT-4.1", "gpt4o": "GPT-4o", "claude_opus": "Claude Opus",
    "claude_sonnet": "Claude Sonnet", "grok3": "Grok-3", "o4_mini": "o4-mini",
    "o3_mini": "o3-mini", "deepseek_r1": "DeepSeek R1", "gemini_flash": "Gemini Flash",
    "gpt4o_mini": "GPT-4o mini", "llama4": "Llama 4 Maverick", "llama33_70b": "Llama 3.3 70B",
    "deepseek_v3": "DeepSeek V3", "qwen3_235b": "Qwen3-235B",
    "mistral": "Mistral Large", "phi4": "Phi-4",
}

MODEL_PARAMS_B = {
    "gpt4_1": 200, "gpt4o": 200, "claude_opus": 200, "claude_sonnet": 70, "grok3": 314,
    "o4_mini": 40, "o3_mini": 40, "deepseek_r1": 37, "gemini_flash": 8, "gpt4o_mini": 8,
    "llama4": 17, "llama33_70b": 70, "deepseek_v3": 37, "qwen3_235b": 22,
    "mistral": 123, "phi4": 14,
}

MODEL_GROUPS = {
    "Frontier Dense": ["gpt4_1", "gpt4o", "claude_opus", "claude_sonnet", "grok3"],
    "Reasoning":      ["o4_mini", "o3_mini", "deepseek_r1"],
    "Efficient":      ["gemini_flash", "gpt4o_mini"],
    "Open Large":     ["llama4", "llama33_70b", "deepseek_v3", "qwen3_235b", "mistral"],
    "Open Small":     ["phi4"],
}

ABLATION_PAIRS = {
    "Reasoning training (same org)":   ("deepseek_r1", "deepseek_v3"),
    "Reasoning training (OpenAI)":     ("o4_mini", "gpt4o_mini"),
    "Model scale (Claude family)":     ("claude_opus", "claude_sonnet"),
    "Model generation (Llama family)": ("llama4", "llama33_70b"),
}

# Hint sensitivity subset (Part B / Cell 15) — 6 representative models to control cost
HINT_TEST_MODELS = ["gpt4o", "claude_opus", "deepseek_r1", "gemini_flash", "llama4", "phi4"]

# ──────────────────────────────────────────────────────────────────────────────
# Part B interpretability model (LOCAL, open-weight, full activation access)
# ──────────────────────────────────────────────────────────────────────────────
# Chosen for: (a) ungated / no HF auth wall, (b) runs comfortably on a free-tier
# T4 GPU in fp16/bf16, (c) instruction-tuned so it follows the same yes/no
# protocol as the API models, (d) small enough that per-layer hooks and
# activation patching are fast to iterate on.
INTERP_MODEL_ID    = "Qwen/Qwen2.5-1.5B-Instruct"
INTERP_MODEL_LABEL = "Qwen2.5-1.5B-Instruct (local, white-box)"
INTERP_N_PROBES    = 400   # probes used for activation extraction
INTERP_HINT_PROBES = 300   # probes used for hint sensitivity

# ──────────────────────────────────────────────────────────────────────────────
# Transformation families
# ──────────────────────────────────────────────────────────────────────────────
TRANSFORMATION_FAMILIES = ["composition", "reversal", "complement", "ordering", "equivalence"]

FAMILY_LABELS = {
    "composition": "Composition", "reversal": "Reversal",
    "complement": "Complement", "ordering": "Ordering",
    "equivalence": "Equivalence",
}

FAMILY_SCORING = {
    "composition": "ljs", "reversal": "rbs_same", "complement": "rbs_opposite",
    "ordering": "rbs_same", "equivalence": "ljs",
}

RBS_RULES = {"reversal": "same", "ordering": "same", "complement": "opposite"}

DOMAINS = ["general", "science", "ethics"]
DIFFICULTIES = ["easy", "medium", "hard"]
DIFFICULTY_DELTA = {"easy": (0.0, 0.35), "medium": (0.35, 0.65), "hard": (0.65, 1.0)}

# ──────────────────────────────────────────────────────────────────────────────
# Cost tracking (USD per 1M tokens, OpenRouter list prices at time of writing)
# ──────────────────────────────────────────────────────────────────────────────
COST_PER_1M = {
    "openai/gpt-4.1": 8.0, "openai/gpt-4o": 7.5, "anthropic/claude-opus-4-5": 22.5,
    "anthropic/claude-sonnet-4-5": 9.0, "x-ai/grok-3": 5.0, "openai/o4-mini": 4.4,
    "openai/o3-mini": 4.4, "deepseek/deepseek-r1": 2.19, "google/gemini-2.0-flash-001": 0.3,
    "openai/gpt-4o-mini": 0.6, "meta-llama/llama-4-maverick": 0.45,
    "meta-llama/llama-3.3-70b-instruct": 0.35, "deepseek/deepseek-v3": 0.9,
    "qwen/qwen3-235b-a22b": 1.5, "mistralai/mistral-large-2411": 4.0,
    "microsoft/phi-4": 0.07, "google/gemini-2.5-pro": 5.0,
}

# ──────────────────────────────────────────────────────────────────────────────
# Run parameters
# ──────────────────────────────────────────────────────────────────────────────
QUICK_TEST       = os.environ.get("CONSISTENCYBENCH_QUICK_TEST", "0") == "1"
N_PER_COMBO      = 5 if QUICK_TEST else 100
TEMPERATURE      = 0.0
MAX_TOKENS_RESP  = 400
MAX_TOKENS_GEN   = 8000
CHECKPOINT_EVERY = 50


def summarize() -> str:
    n_probes_est = N_PER_COMBO * len(TRANSFORMATION_FAMILIES) * len(DOMAINS) * len(DIFFICULTIES)
    n_api_calls = n_probes_est * len(MODELS) * 2
    lines = [
        "=" * 68,
        "  ConsistencyBench — NeurIPS 2026 Configuration",
        "=" * 68,
        f"  Generator (excluded from eval): {GENERATOR_MODEL}",
        f"  Evaluated API models: {len(MODELS)}",
    ]
    for g, ms in MODEL_GROUPS.items():
        lines.append(f"    [{g}]: {', '.join(ms)}")
    lines += [
        f"  Interpretability model (local): {INTERP_MODEL_ID}",
        f"  Transformation families: {TRANSFORMATION_FAMILIES}",
        f"  Probes/combo: {N_PER_COMBO}  |  Total probes: {n_probes_est:,}",
        f"  Total API calls (Part A): {n_api_calls:,}",
        f"  Data directory: {BASE_DIR}",
        "=" * 68,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ensure_dirs()
    print(summarize())
