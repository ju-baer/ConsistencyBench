"""
Part B, Stage 1 — Activation Extraction on a Local Open-Weight Model.

Loads `Qwen2.5-1.5B-Instruct` locally with full white-box access via
`LocalHFBackend`, runs it through a balanced sample of the probe set using the
same harness and system prompt as every API model (so its behavioral numbers
are directly comparable), and captures the full residual-stream activation
(every layer's hidden state at the final prompt token) at the moment the model
begins generating its answer for both prompt_a and prompt_b of each probe.

This is the representation the model is actually using to decide, and it's
exactly the representation `shortcut_probe.py` and `causal_tests.py` analyze
and intervene on.

Direct, environment-independent port of Notebook Cell 16.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .. import config
from ..client import load_checkpoint, save_checkpoint
from ..harness import LocalHFBackend
from ..scoring import score_result

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the following question directly "
    "and concisely. When the question has a clear yes or no answer, begin your "
    "response with 'Yes' or 'No' followed by a brief explanation of 1-3 sentences. "
    "Do not hedge unnecessarily."
)


def sample_balanced_probes(all_probes: list[dict], n_probes: int = config.INTERP_N_PROBES) -> list[dict]:
    random.shuffle(all_probes)
    per_family_target = n_probes // len(config.TRANSFORMATION_FAMILIES)
    interp_probes = []
    for fam in config.TRANSFORMATION_FAMILIES:
        fam_probes = [p for p in all_probes if p["family"] == fam]
        interp_probes.extend(fam_probes[:per_family_target])
    return interp_probes


def extract_activations(
    all_probes: list[dict],
    n_probes: int = config.INTERP_N_PROBES,
    local_dir: str | Path = "local_activations",
    seed: int = 11,
) -> tuple[LocalHFBackend, list[dict], Path]:
    """Load the local model and extract per-layer residual-stream activations
    for a balanced sample of probes. Returns (backend, interp_probes, local_dir)."""
    random.seed(seed)
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    act_dir = config.BASE_DIR / "interpretability" / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    interp_backend = LocalHFBackend(config.INTERP_MODEL_ID)

    interp_probes = sample_balanced_probes(all_probes, n_probes)
    print(f"Sampled {len(interp_probes)} probes for activation extraction "
          f"({n_probes // len(config.TRANSFORMATION_FAMILIES)} per family)")

    interp_raw = load_checkpoint("cb_interp_raw.json") or []
    done_interp = {r["probe_id"] for r in interp_raw}

    for probe in tqdm(interp_probes, desc="Extracting activations"):
        pid = probe["probe_id"]
        if pid in done_interp:
            continue
        resp_a, acts_a = interp_backend.query_with_activations(probe["prompt_a"], system=SYSTEM_PROMPT)
        resp_b, acts_b = interp_backend.query_with_activations(probe["prompt_b"], system=SYSTEM_PROMPT)

        # acts_a/acts_b: list of length N_LAYERS+1, each np.array[hidden_size]
        np.save(local_dir / f"{pid}_a.npy", np.stack(acts_a))
        np.save(local_dir / f"{pid}_b.npy", np.stack(acts_b))

        interp_raw.append({
            "probe_id": pid, "family": probe["family"], "domain": probe["domain"],
            "difficulty": probe["difficulty"], "delta": probe.get("delta", 0.0),
            "model": config.INTERP_MODEL_ID, "model_id": config.INTERP_MODEL_ID,
            "prompt_a": probe["prompt_a"], "prompt_b": probe["prompt_b"],
            "logical_constraint": probe.get("logical_constraint", ""),
            "expected_inconsistency": probe.get("expected_inconsistency", ""),
            "scoring_hint": probe.get("scoring_hint", config.FAMILY_SCORING.get(probe["family"], "ljs")),
            "response_a": resp_a, "response_b": resp_b,
            "error_a": None, "error_b": None,
        })
        if len(interp_raw) % 25 == 0:
            save_checkpoint(interp_raw, "cb_interp_raw.json")

    save_checkpoint(interp_raw, "cb_interp_raw.json")
    os.system(f"cp -r {local_dir}/* {act_dir}/ 2>/dev/null")

    print(f"\nActivation extraction complete: {len(interp_raw)} probes x 2 prompts")
    print(f"Layers captured: {interp_backend.n_layers + 1} "
          f"(embedding + {interp_backend.n_layers} transformer layers)")

    return interp_backend, interp_probes, local_dir


def score_local_model(interp_raw: list[dict]) -> list[dict]:
    """Score the local model's own responses with the same RBS+LJS ensemble
    used for the 16 API models."""
    interp_scored = load_checkpoint("cb_interp_scored.json") or []
    interp_scored_ids = {r["probe_id"] for r in interp_scored}
    to_score = [r for r in interp_raw if r["probe_id"] not in interp_scored_ids]
    print(f"\nScoring {len(to_score)} local-model responses...")

    for result in tqdm(to_score, desc="Scoring local model"):
        scored = score_result(result, judge=config.GENERATOR_MODEL)
        interp_scored.append(scored)
        if scored.get("score_method") == "ljs":
            time.sleep(0.4)
        if len(interp_scored) % 25 == 0:
            save_checkpoint(interp_scored, "cb_interp_scored.json")
    save_checkpoint(interp_scored, "cb_interp_scored.json")

    scoreable = [r for r in interp_scored if r.get("consistent") is not None]
    ir = 100 * sum(1 for r in scoreable if not r["consistent"]) / max(len(scoreable), 1)
    print(f"\n{config.INTERP_MODEL_LABEL}")
    print(f"  Scored pairs: {len(scoreable)} | Overall IR: {ir:.1f}%")
    print("  (Reference point only — this model is the interpretability testbed, not a "
          "leaderboard entry: it is far smaller than the 16 evaluated models.)")
    return interp_scored


def build_activation_matrix(
    interp_scored: list[dict], n_layers: int, local_dir: str | Path = "local_activations",
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, list[int]]:
    """Build the labeled activation matrix for prompt_b (the "decision"
    representation). Returns (X_by_layer, y, fam_arr, valid_ids) where
    X_by_layer[l] has shape [n_probes, hidden_size] and y is 1=inconsistent."""
    local_dir = Path(local_dir)
    interp_scoreable = [r for r in interp_scored if r.get("consistent") is not None]

    interp_labels, interp_probe_ids, interp_families = [], [], []
    for r in interp_scoreable:
        interp_labels.append(int(not r["consistent"]))
        interp_probe_ids.append(r["probe_id"])
        interp_families.append(r["family"])

    X_by_layer: dict[int, list] = {l: [] for l in range(n_layers + 1)}
    valid_ids = []
    for pid in interp_probe_ids:
        path_b = local_dir / f"{pid}_b.npy"
        if not path_b.exists():
            continue
        acts_b = np.load(path_b)  # shape [N_LAYERS+1, hidden_size]
        for l in range(n_layers + 1):
            X_by_layer[l].append(acts_b[l])
        valid_ids.append(pid)

    id_to_label = dict(zip(interp_probe_ids, interp_labels))
    id_to_family = dict(zip(interp_probe_ids, interp_families))
    y = np.array([id_to_label[pid] for pid in valid_ids])
    fam_arr = np.array([id_to_family[pid] for pid in valid_ids])
    X_by_layer_arr = {l: np.stack(v) for l, v in X_by_layer.items()}

    print(f"\nBuilt activation matrix: {len(valid_ids)} examples x {n_layers + 1} layers "
          f"x {X_by_layer_arr[0].shape[1]} hidden dims")
    print(f"Label balance: {y.sum()} inconsistent / {len(y) - y.sum()} consistent "
          f"({100 * y.mean():.1f}% inconsistent)")

    out_path = config.BASE_DIR / "interpretability" / "activation_matrix.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, y=y, families=fam_arr, valid_ids=np.array(valid_ids),
              **{f"layer_{l}": X_by_layer_arr[l] for l in X_by_layer_arr})
    print(f"Saved activation matrix to {out_path}")

    return X_by_layer_arr, y, fam_arr, valid_ids
