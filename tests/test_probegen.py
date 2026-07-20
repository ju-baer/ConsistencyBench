"""Tests for the pure-logic parts of consistencybench.probegen: constraint-spec
construction and trigram-Jaccard deduplication. Generation itself requires an
API call and is not covered by unit tests."""
from consistencybench.probegen import build_constraint_spec, deduplicate_probes


def test_build_constraint_spec_uses_the_right_delta_band():
    spec = build_constraint_spec("complement", "ethics", "hard")
    assert spec["family"] == "complement"
    assert spec["domain"] == "ethics"
    assert spec["delta_target"]["min"] == 0.65
    assert spec["delta_target"]["max"] == 1.0


def test_deduplicate_probes_removes_near_duplicates():
    probes = [
        {"prompt_a": "Is the sky blue on a clear day?"},
        {"prompt_a": "Is the sky blue on a clear day?!"},  # trivial punctuation-only variant
        {"prompt_a": "Did the Roman Empire fall before the Renaissance began?"},
    ]
    unique = deduplicate_probes(probes, threshold=0.82)
    assert len(unique) == 2


def test_deduplicate_probes_keeps_genuinely_different_probes():
    probes = [
        {"prompt_a": "Is water wet?"},
        {"prompt_a": "Did the French Revolution precede the Industrial Revolution?"},
        {"prompt_a": "Should hospitals disclose treatment risks to patients?"},
    ]
    unique = deduplicate_probes(probes)
    assert len(unique) == 3


def test_deduplicate_probes_handles_empty_list():
    assert deduplicate_probes([]) == []
