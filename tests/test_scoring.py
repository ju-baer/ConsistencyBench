"""Tests for the deterministic parts of consistencybench.scoring: yes/no
extraction and Rule-Based Scoring. LJS (LLM-as-Judge) scoring requires network
access and is not covered by unit tests."""
import pytest

from consistencybench.scoring import extract_yn, rule_based_score


@pytest.mark.parametrize("text,expected", [
    ("Yes, that is correct.", "yes"),
    ("yes.", "yes"),
    ("No, that is not the case.", "no"),
    ("Certainly, this follows.", "yes"),
    ("Never has this been true.", "no"),
    ("It depends on the context.", None),
    ("", None),
    (None, None),
])
def test_extract_yn(text, expected):
    assert extract_yn(text) == expected


def test_rule_based_score_same_rule_consistent():
    result = {"family": "reversal", "response_a": "Yes, X relates to Y.", "response_b": "Yes, Y relates to X."}
    scored = rule_based_score(result)
    assert scored is not None
    assert scored["consistent"] is True
    assert scored["ans_a"] == "yes" and scored["ans_b"] == "yes"


def test_rule_based_score_same_rule_inconsistent():
    result = {"family": "ordering", "response_a": "Yes, A came first.", "response_b": "No, B did not come after."}
    scored = rule_based_score(result)
    assert scored is not None
    assert scored["consistent"] is False


def test_rule_based_score_opposite_rule_consistent():
    # complement: consistent means the two answers DIFFER (P and NOT-P can't both be affirmed)
    result = {"family": "complement", "response_a": "Yes, P holds.", "response_b": "No, NOT-P does not hold."}
    scored = rule_based_score(result)
    assert scored is not None
    assert scored["consistent"] is True


def test_rule_based_score_opposite_rule_inconsistent():
    result = {"family": "complement", "response_a": "Yes, P holds.", "response_b": "Yes, NOT-P also holds."}
    scored = rule_based_score(result)
    assert scored is not None
    assert scored["consistent"] is False


def test_rule_based_score_returns_none_for_ljs_families():
    result = {"family": "composition", "response_a": "Yes.", "response_b": "No."}
    assert rule_based_score(result) is None


def test_rule_based_score_returns_none_when_unparseable():
    result = {"family": "reversal", "response_a": "It's complicated.", "response_b": "Yes."}
    assert rule_based_score(result) is None
