"""
OpenRouter client + utilities: retrying API client, ASCII-safe headers,
bracket-depth JSON recovery for truncated generations, and checkpointing so any
stage of the pipeline can resume after an interruption without losing progress
or re-spending API budget.

Direct, environment-independent port of Notebook Cell 3.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from . import config

# ──────────────────────────────────────────────────────────────────────────────
# Cost tracking (module-level, mutated by call_model)
# ──────────────────────────────────────────────────────────────────────────────
total_cost_usd = 0.0


def _ascii(s: str) -> str:
    """NFKD normalize + ASCII encode. Prevents UnicodeEncodeError in HTTP headers."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def get_client() -> OpenAI:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it or add it to a .env file "
            "before calling any function that queries the API."
        )
    return OpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/YOUR_USERNAME/ConsistencyBench",
            "X-Title": _ascii("ConsistencyBench - NeurIPS 2026"),
        },
    )


_client: OpenAI | None = None


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(Exception),
)
def call_model(
    model_id: str,
    prompt: str,
    system: str = "",
    max_tokens: int = config.MAX_TOKENS_RESP,
    temperature: float = config.TEMPERATURE,
) -> str:
    """Query a model via OpenRouter and return its text response.

    Retries with exponential backoff on any exception (rate limits, transient
    5xx errors, etc.). Accumulates a running USD cost estimate in
    `total_cost_usd` based on `config.COST_PER_1M`.
    """
    global _client, total_cost_usd
    if _client is None:
        _client = get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = _client.chat.completions.create(
        model=model_id, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    if resp.usage:
        tok = (resp.usage.prompt_tokens + resp.usage.completion_tokens) / 1_000_000
        total_cost_usd += tok * config.COST_PER_1M.get(model_id, 5.0)
    return resp.choices[0].message.content.strip()


def safe_json(text: str) -> list[Any]:
    """Bracket-depth JSON extraction: handles truncated arrays, single objects,
    and markdown code fences that LLMs commonly wrap JSON output in."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    def extract(s: str, oc: str, cc: str) -> str | None:
        start = s.find(oc)
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(s[start:], start):
            if ch == oc:
                depth += 1
            elif ch == cc:
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        return s[start:] + cc * depth

    block = extract(text, "[", "]") or extract(text, "{", "}")
    if block is None:
        raise ValueError(f"No JSON found: {text[:200]!r}")
    try:
        parsed = json.loads(block)
    except json.JSONDecodeError:
        last = block.rfind("},")
        if last != -1:
            block = block[:last + 1]
        opens = block.count("[") - block.count("]")
        block = block + "]" * max(0, opens)
        parsed = json.loads(block)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list, got {type(parsed)}")
    return parsed


def save_checkpoint(data: Any, filename: str, base_dir: Path = config.BASE_DIR) -> None:
    """Write a checkpoint both to the given filename and to base_dir/checkpoints/."""
    drive_path = base_dir / "checkpoints" / os.path.basename(filename)
    drive_path.parent.mkdir(parents=True, exist_ok=True)
    for path in [Path(filename), drive_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def load_checkpoint(filename: str, base_dir: Path = config.BASE_DIR) -> Any | None:
    """Load a checkpoint, preferring base_dir/checkpoints/ over the raw filename."""
    drive_path = base_dir / "checkpoints" / os.path.basename(filename)
    for path in [drive_path, Path(filename)]:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            print(f"Loaded checkpoint: {path} ({len(data)} items)")
            return data
    return None


def sanitize_probe(p: dict) -> dict:
    return {
        k: v.encode("ascii", "ignore").decode("ascii") if isinstance(v, str) else v
        for k, v in p.items()
    }
