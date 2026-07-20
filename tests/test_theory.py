"""Tests for consistencybench.theory that don't require downloading the
sentence-embedding model (classify_difficulty, and the formal-spec table)."""
import pytest

from consistencybench.theory import FAMILY_FORMAL_SPEC, classify_difficulty


@pytest.mark.parametrize("delta,expected", [
    (0.0, "easy"), (0.1, "easy"), (0.349, "easy"),
    (0.35, "medium"), (0.5, "medium"), (0.649, "medium"),
    (0.65, "hard"), (0.9, "hard"), (1.0, "hard"),
])
def test_classify_difficulty_boundaries(delta, expected):
    assert classify_difficulty(delta) == expected


def test_every_family_has_a_complete_formal_spec():
    required_keys = {
        "name", "logical_basis", "formal_constraint", "consistency_rule",
        "probe_structure", "scoring", "real_world_coverage_pct",
    }
    for fam, spec in FAMILY_FORMAL_SPEC.items():
        assert required_keys.issubset(spec.keys()), f"{fam} spec missing keys"
        assert "prompt_a" in spec["probe_structure"]
        assert "prompt_b" in spec["probe_structure"]
        assert spec["scoring"] in ("ljs", "rbs_same", "rbs_opposite")


def test_family_coverage_sums_to_a_plausible_total():
    total = sum(s["real_world_coverage_pct"] for s in FAMILY_FORMAL_SPEC.values())
    # The five families are documented to cover ~83% of real-world logical
    # relationships, with the remainder attributed to multi-hop/modal/quantifier
    # reasoning outside this benchmark's scope (see README methodology section).
    assert 75.0 < total < 90.0
