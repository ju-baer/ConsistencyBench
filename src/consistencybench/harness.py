"""
Reproducible Evaluation Harness.

Every experiment queries models through a unified `ModelBackend` interface,
implemented once for the OpenRouter API (`APIBackend`) and once for local
open-weight models loaded via `transformers` (`LocalHFBackend`). The harness
that drives both (`EvaluationHarness`) is identical in either case, so any
behavioral difference observed between the 16 API models and the local
interpretability model is a genuine model difference, not an artifact of two
different evaluation code paths.

A `run_id` is derived from the full run configuration (backend, model, system
prompt, generation params) so that results from different configurations are
never silently mixed in the same checkpoint file — a change to any of these
produces a new run_id and therefore a new checkpoint namespace.

Direct, environment-independent port of Notebook Cell 7.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from . import config
from .client import call_model, load_checkpoint, save_checkpoint


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# Backend interface
# ──────────────────────────────────────────────────────────────────────────────
class ModelBackend(ABC):
    """Unified interface a model must satisfy to be evaluated by the harness.
    Both the OpenRouter API and any local transformers model implement this,
    so probe-querying logic never has to know or care which one it's talking to."""

    backend_type: str  # "api" or "local_hf"
    model_id: str

    @abstractmethod
    def query(self, prompt: str, system: str = "", max_tokens: int = 400) -> dict:
        """Returns {'response': str|None, 'latency_s': float|None, 'error': str|None}."""
        ...

    def config_dict(self) -> dict:
        return {"backend_type": self.backend_type, "model_id": self.model_id}


class APIBackend(ModelBackend):
    """Wraps the OpenRouter `call_model` function."""
    backend_type = "api"

    def __init__(self, model_id: str):
        self.model_id = model_id

    def query(self, prompt: str, system: str = "", max_tokens: int = 400) -> dict:
        t0 = time.time()
        try:
            r = call_model(self.model_id, prompt, system=system, max_tokens=max_tokens)
            return {"response": r, "latency_s": round(time.time() - t0, 2), "error": None}
        except Exception as e:
            return {"response": None, "latency_s": None, "error": str(e)}


class LocalHFBackend(ModelBackend):
    """Loads an open-weight model locally via `transformers` with full activation
    access. Generation is deterministic (greedy decoding) to mirror
    temperature=0 on the API side."""
    backend_type = "local_hf"

    def __init__(self, model_id: str, device: Optional[str] = None, dtype=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)

        print(f"Loading {model_id} on {self.device} ({self.dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=self.dtype, device_map=self.device,
            output_hidden_states=True,
        )
        self.model.eval()
        self.n_layers = self.model.config.num_hidden_layers
        print(f"Loaded. {self.n_layers} layers, hidden_size={self.model.config.hidden_size}")

    def _build_chat_input(self, prompt: str, system: str = ""):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def query(self, prompt: str, system: str = "", max_tokens: int = 400) -> dict:
        t0 = time.time()
        try:
            inputs = self._build_chat_input(prompt, system)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=False,          # greedy = deterministic, mirrors temperature=0
                    temperature=None, top_p=None, top_k=None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_tokens = out[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            return {"response": response, "latency_s": round(time.time() - t0, 2), "error": None}
        except Exception as e:
            return {"response": None, "latency_s": None, "error": str(e)}

    def query_with_activations(self, prompt: str, system: str = "", max_tokens: int = 400):
        """Like `query`, but also returns per-layer residual-stream activations for
        the *final prompt token* at generation start — the representation the
        model uses to decide its first output token. Used by the Part B
        interpretability pipeline (activations, shortcut_probe, causal_tests)."""
        inputs = self._build_chat_input(prompt, system)
        with torch.no_grad():
            fwd = self.model(**inputs, output_hidden_states=True)
            # hidden_states: tuple of (n_layers+1) tensors, each [batch, seq, hidden]
            last_token_acts = [h[0, -1, :].float().cpu().numpy() for h in fwd.hidden_states]
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen_tokens = out[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return response, last_token_acts  # list of np.array[hidden_size], len = n_layers+1


# ──────────────────────────────────────────────────────────────────────────────
# The harness itself
# ──────────────────────────────────────────────────────────────────────────────
class EvaluationHarness:
    """Runs a backend against a probe set with deterministic, versioned, resumable
    execution."""

    def __init__(
        self, backend: ModelBackend, system_prompt: str,
        checkpoint_dir: str | Path, drive_base: str | Path = config.BASE_DIR, seed: int = 42,
    ):
        self.backend = backend
        self.system_prompt = system_prompt
        self.checkpoint_dir = Path(checkpoint_dir)
        self.drive_base = Path(drive_base)
        self.seed = seed
        set_global_seed(seed)
        self.run_id = self._compute_run_id()

    def _compute_run_id(self) -> str:
        cfg = {**self.backend.config_dict(),
               "system_prompt": self.system_prompt, "seed": self.seed}
        cfg_str = json.dumps(cfg, sort_keys=True)
        return hashlib.sha256(cfg_str.encode()).hexdigest()[:12]

    def checkpoint_path(self, tag: str) -> Path:
        return self.checkpoint_dir / f"harness_{tag}_{self.run_id}.json"

    def run(
        self, probes: list, delay: float = 0.4, checkpoint_every: int = 50,
        resume: bool = True, tag: str = "run",
    ) -> list:
        ckpt_path = self.checkpoint_path(tag)
        results = (load_checkpoint(str(ckpt_path)) if resume else None) or []
        done = {r["probe_id"] for r in results}
        todo = [p for p in probes if p["probe_id"] not in done]

        print(f"[Harness run_id={self.run_id}] backend={self.backend.backend_type} "
              f"model={self.backend.model_id}")
        print(f"  Total probes: {len(probes)} | Done: {len(done)} | Remaining: {len(todo)}")

        for i, probe in enumerate(tqdm(todo, desc=f"Harness[{self.backend.model_id}]")):
            ra = self.backend.query(probe["prompt_a"], system=self.system_prompt)
            time.sleep(delay)
            rb = self.backend.query(probe["prompt_b"], system=self.system_prompt)
            time.sleep(delay)
            results.append({
                "probe_id": probe["probe_id"], "family": probe["family"],
                "domain": probe["domain"], "difficulty": probe["difficulty"],
                "delta": probe.get("delta", 0.0),
                "model": self.backend.model_id, "model_id": self.backend.model_id,
                "backend_type": self.backend.backend_type, "run_id": self.run_id,
                "prompt_a": probe["prompt_a"], "prompt_b": probe["prompt_b"],
                "logical_constraint": probe.get("logical_constraint", ""),
                "expected_inconsistency": probe.get("expected_inconsistency", ""),
                "scoring_hint": probe.get("scoring_hint", config.FAMILY_SCORING.get(probe["family"], "ljs")),
                "response_a": ra["response"], "latency_a": ra["latency_s"], "error_a": ra["error"],
                "response_b": rb["response"], "latency_b": rb["latency_s"], "error_b": rb["error"],
            })
            if (i + 1) % checkpoint_every == 0:
                save_checkpoint(results, str(ckpt_path))

        save_checkpoint(results, str(ckpt_path))
        return results
