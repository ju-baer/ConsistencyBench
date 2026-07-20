"""Tests for consistencybench.client's robust JSON parsing -- the part of the
pipeline most exposed to messy, real-world LLM output (markdown fences,
truncated arrays, stray prose)."""
import pytest

from consistencybench.client import safe_json, sanitize_probe


def test_safe_json_parses_clean_array():
    text = '[{"a": 1}, {"a": 2}]'
    assert safe_json(text) == [{"a": 1}, {"a": 2}]


def test_safe_json_strips_markdown_fences():
    text = '```json\n[{"a": 1}]\n```'
    assert safe_json(text) == [{"a": 1}]


def test_safe_json_wraps_single_object_in_a_list():
    text = '{"a": 1}'
    assert safe_json(text) == [{"a": 1}]


def test_safe_json_handles_truncated_array():
    # A response cut off mid-object (e.g. hit max_tokens) should still recover
    # the objects that completed before the cut.
    text = '[{"a": 1}, {"a": 2}, {"a": 3'
    result = safe_json(text)
    assert result[0] == {"a": 1}
    assert result[1] == {"a": 2}


def test_safe_json_raises_on_no_json_found():
    with pytest.raises(ValueError):
        safe_json("This is just prose with no JSON in it at all.")


def test_safe_json_strips_non_ascii():
    # NFKD-normalizing "café" decomposes é into e + combining acute accent;
    # ascii-encode("ignore") then drops the (non-ASCII) combining accent,
    # leaving the base letter behind.
    text = '[{"a": "caf\u00e9"}]'
    result = safe_json(text)
    assert result[0]["a"] == "cafe"


def test_sanitize_probe_only_touches_strings():
    probe = {"prompt_a": "caf\u00e9", "probe_id": 7, "delta": 0.5}
    sanitized = sanitize_probe(probe)
    assert sanitized["prompt_a"] == "caf"
    assert sanitized["probe_id"] == 7
    assert sanitized["delta"] == 0.5
