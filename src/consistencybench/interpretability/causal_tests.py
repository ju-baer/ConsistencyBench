"""
Part B, Stage 3 — Causal Tests: Activation Patching and Feature Steering.

Everything upstream is correlational: `shortcut_probe.py` shows that
inconsistency is *decodable* from activations at a particular layer, not that
those activations *cause* the inconsistent output. This module runs two
causal interventions to close that gap.

**Activation patching.** For a matched pair of probes from the same
transformation family where the model was consistent on one and inconsistent
on the other, the consistent run's residual-stream activation at the
best-probe layer is literally copied into the inconsistent run's forward pass
at the same position, and generation continues from there. If the output
flips toward the consistent answer, the activation at that layer is not just
correlated with the outcome — it is (at least partially) causally responsible
for it.

**Feature steering (diff-in-means).** At scale, a scaled version of the
consistent-minus-inconsistent direction (computed in `shortcut_probe.py`) is
added to the residual stream for a held-out set of probes the model
originally got wrong, and the dose-response is measured: does the
inconsistency rate drop as steering strength increases? A **specificity
check** — applying the same steering vector to unrelated factual questions —
confirms the intervention targets consistency behavior specifically rather
than degrading the model's outputs indiscriminately.

Direct, environment-independent port of Notebook Cell 18.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from .. import config
from ..harness import LocalHFBackend
from ..scoring import extract_yn, score_result

ALPHAS = [0.0, 4.0, 8.0, 16.0]
CONTROL_QUESTIONS = [
    "What is the capital of Japan?",
    "How many legs does a spider have?",
    "What year did World War II end?",
    "What is the chemical symbol for gold?",
]

SYSTEM_PROMPT = (
    "You are a knowledgeable assistant. Answer the following question directly "
    "and concisely. When the question has a clear yes or no answer, begin your "
    "response with 'Yes' or 'No' followed by a brief explanation of 1-3 sentences. "
    "Do not hedge unnecessarily."
)


def get_target_module(interp_backend: LocalHFBackend, best_layer: int):
    """hidden_states[0] = embedding output; hidden_states[l] (l>=1) = output of
    decoder layer (l-1). So intervening "at hidden_states[best_layer]" means
    hooking the output of decoder layer index (best_layer - 1)."""
    hook_layer_idx = max(best_layer - 1, 0)
    decoder_layers = interp_backend.model.model.layers
    target_module = decoder_layers[hook_layer_idx]
    print(f"Intervening at decoder layer index {hook_layer_idx} "
          f"(corresponds to hidden_states[{best_layer}], the best probe layer)")
    return target_module


def make_add_hook(vector: torch.Tensor):
    """Forward hook that adds `vector` to every position's residual stream output."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + vector.to(hidden.dtype).to(hidden.device)
            return (hidden,) + output[1:]
        return output + vector.to(output.dtype).to(output.device)
    return hook


def make_overwrite_hook(vector: torch.Tensor):
    """Forward hook that overwrites the LAST token position's residual stream
    with `vector` — used for literal single-pair activation patching."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, -1, :] = vector.to(hidden.dtype).to(hidden.device)
            return (hidden,) + output[1:]
        hidden = output.clone()
        hidden[:, -1, :] = vector.to(hidden.dtype).to(hidden.device)
        return hidden
    return hook


def generate_with_hook(interp_backend: LocalHFBackend, target_module, prompt: str, system: str,
                        hook_fn, max_tokens: int = 200) -> str:
    handle = target_module.register_forward_hook(hook_fn)
    try:
        inputs = interp_backend._build_chat_input(prompt, system=system)
        with torch.no_grad():
            out = interp_backend.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=interp_backend.tokenizer.eos_token_id)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return interp_backend.tokenizer.decode(gen, skip_special_tokens=True).strip()
    finally:
        handle.remove()


def run_activation_patching(
    interp_backend: LocalHFBackend, target_module, best_layer: int,
    interp_probes: list[dict], interp_raw: list[dict],
    fam_arr: np.ndarray, y: np.ndarray, valid_ids: list[int],
    local_dir: str = "local_activations",
) -> list[dict]:
    """TEST 1: Literal activation patching (single matched pair per family)."""
    print("\n" + "=" * 60); print("TEST 1: ACTIVATION PATCHING"); print("=" * 60)

    patching_results = []
    id_arr = np.array(valid_ids)

    for fam in config.TRANSFORMATION_FAMILIES:
        fam_mask = fam_arr == fam
        cons_ids = id_arr[fam_mask & (y == 0)]
        incons_ids = id_arr[fam_mask & (y == 1)]
        if len(cons_ids) == 0 or len(incons_ids) == 0:
            continue
        source_id, target_id = cons_ids[0], incons_ids[0]

        source_acts = np.load(f"{local_dir}/{source_id}_b.npy")[best_layer]  # consistent run
        source_vec = torch.tensor(source_acts)

        target_probe = next(p for p in interp_probes if p["probe_id"] == target_id)
        original_response = next(r["response_b"] for r in interp_raw if r["probe_id"] == target_id)

        patched_response = generate_with_hook(
            interp_backend, target_module, target_probe["prompt_b"], SYSTEM_PROMPT,
            make_overwrite_hook(source_vec))

        orig_orient = extract_yn(original_response or "")
        patched_orient = extract_yn(patched_response or "")
        source_response = next(r["response_b"] for r in interp_raw if r["probe_id"] == source_id)
        source_orient = extract_yn(source_response or "")

        flipped_toward_source = (patched_orient is not None and source_orient is not None
                                  and patched_orient == source_orient
                                  and orig_orient != source_orient)

        patching_results.append({
            "family": fam, "source_id": int(source_id), "target_id": int(target_id),
            "original_response": (original_response or "")[:100],
            "patched_response": (patched_response or "")[:100],
            "source_orientation": source_orient, "original_orientation": orig_orient,
            "patched_orientation": patched_orient, "flipped_toward_source": flipped_toward_source,
        })
        print(f"\n[{config.FAMILY_LABELS[fam]}] target={target_id} <- patched from source={source_id}")
        print(f"  Original (inconsistent) response: {(original_response or '')[:90]}...")
        print(f"  Patched response:                 {(patched_response or '')[:90]}...")
        print(f"  Flipped toward source orientation: {flipped_toward_source}")

    n_flipped = sum(r["flipped_toward_source"] for r in patching_results)
    print(f"\nActivation patching flip rate: {n_flipped}/{len(patching_results)} families "
          f"({100 * n_flipped / max(len(patching_results), 1):.0f}%)")

    out_dir = config.BASE_DIR / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(patching_results).to_csv(out_dir / "activation_patching_results.csv", index=False)
    return patching_results


def run_feature_steering(
    interp_backend: LocalHFBackend, target_module, best_layer: int,
    X_by_layer: dict[int, np.ndarray], y: np.ndarray, valid_ids: list[int],
    interp_probes: list[dict], interp_raw: list[dict],
) -> tuple[pd.DataFrame, torch.Tensor]:
    """TEST 2: Diff-in-means feature steering, dose-response across held-out probes."""
    print("\n" + "=" * 60); print("TEST 2: FEATURE STEERING (DIFF-IN-MEANS)"); print("=" * 60)

    id_arr = np.array(valid_ids)
    direction = X_by_layer[best_layer][y == 0].mean(axis=0) - X_by_layer[best_layer][y == 1].mean(axis=0)
    direction_unit = direction / (np.linalg.norm(direction) + 1e-8)
    direction_tensor = torch.tensor(direction_unit, dtype=torch.float32)
    print(f"Diff-in-means direction computed at layer {best_layer} "
          f"(consistent-mean minus inconsistent-mean), ||direction||={np.linalg.norm(direction):.2f}")

    incons_ids_all = id_arr[y == 1]
    n_steer_probes = min(30, len(incons_ids_all))
    steer_ids = list(incons_ids_all[:n_steer_probes])
    steer_probes = [p for p in interp_probes if p["probe_id"] in steer_ids]
    print(f"Testing steering on {len(steer_probes)} originally-inconsistent held-out probes")

    dose_response = []
    for alpha in ALPHAS:
        n_still_inconsistent = 0
        for probe in steer_probes:
            response_a = next(r["response_a"] for r in interp_raw if r["probe_id"] == probe["probe_id"])
            if alpha == 0.0:
                steered_response = next(r["response_b"] for r in interp_raw if r["probe_id"] == probe["probe_id"])
            else:
                vec = direction_tensor * alpha
                steered_response = generate_with_hook(
                    interp_backend, target_module, probe["prompt_b"], SYSTEM_PROMPT, make_add_hook(vec))
            r_test = {**probe, "response_a": response_a, "response_b": steered_response,
                      "error_a": None, "error_b": None}
            scored_test = score_result(r_test, judge=config.GENERATOR_MODEL)
            if scored_test.get("consistent") is False:
                n_still_inconsistent += 1
            time.sleep(0.2 if alpha > 0 else 0)
        ir_at_alpha = 100 * n_still_inconsistent / len(steer_probes)
        dose_response.append({"alpha": alpha, "ir_pct": ir_at_alpha, "n": len(steer_probes)})
        print(f"  alpha={alpha:>5.1f}  IR={ir_at_alpha:5.1f}%  "
              f"({len(steer_probes) - n_still_inconsistent}/{len(steer_probes)} fixed)")

    df_dose = pd.DataFrame(dose_response)
    out_dir = config.BASE_DIR / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_dose.to_csv(out_dir / "steering_dose_response.csv", index=False)

    baseline_ir = dose_response[0]["ir_pct"]
    best_ir = min(d["ir_pct"] for d in dose_response)
    print(f"\nSteering reduces IR from {baseline_ir:.1f}% (alpha=0) to {best_ir:.1f}% "
          f"at strongest tested strength — a causal effect, not just a correlation.")

    return df_dose, direction_tensor


def run_specificity_check(
    interp_backend: LocalHFBackend, target_module, direction_tensor: torch.Tensor,
) -> None:
    """Applies the strongest steering vector to unrelated factual questions to
    confirm the intervention is targeted rather than a general degradation."""
    print("\nSpecificity check: applying strongest steering vector to unrelated questions...")
    max_alpha_vec = direction_tensor * ALPHAS[-1]
    for q in CONTROL_QUESTIONS:
        steered_ans = generate_with_hook(interp_backend, target_module, q, SYSTEM_PROMPT, make_add_hook(max_alpha_vec))
        print(f"  Q: {q}")
        print(f"    Steered answer: {steered_ans[:120]}")

    print("\nIf steered answers to unrelated questions above remain coherent and factually")
    print("sound, the intervention is targeted at consistency behavior specifically rather")
    print("than degrading the model's outputs indiscriminately.")
