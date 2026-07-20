"""Sanity checks on the static configuration: every model has a label, every
family has consistent scoring metadata, and the ablation pairs reference real
models. These catch typos/drift in config.py without needing network access.
"""
from consistencybench import config


def test_every_model_has_a_label():
    for key in config.MODELS:
        assert key in config.MODEL_LABELS, f"{key} missing from MODEL_LABELS"
        assert key in config.MODEL_PARAMS_B, f"{key} missing from MODEL_PARAMS_B"


def test_every_model_in_exactly_one_group():
    grouped = [m for members in config.MODEL_GROUPS.values() for m in members]
    assert sorted(grouped) == sorted(config.MODELS.keys())
    assert len(grouped) == len(set(grouped)), "a model appears in more than one group"


def test_ablation_pairs_reference_real_models():
    for name, (m1, m2) in config.ABLATION_PAIRS.items():
        assert m1 in config.MODELS, f"{name}: {m1} not in MODELS"
        assert m2 in config.MODELS, f"{name}: {m2} not in MODELS"


def test_hint_test_models_are_real_models():
    for m in config.HINT_TEST_MODELS:
        assert m in config.MODELS


def test_family_scoring_covers_every_family():
    for fam in config.TRANSFORMATION_FAMILIES:
        assert fam in config.FAMILY_SCORING
        assert config.FAMILY_SCORING[fam] in ("ljs", "rbs_same", "rbs_opposite")


def test_rbs_rules_are_same_or_opposite():
    for fam, rule in config.RBS_RULES.items():
        assert fam in config.TRANSFORMATION_FAMILIES
        assert rule in ("same", "opposite")


def test_difficulty_bands_are_contiguous_and_cover_0_to_1():
    bands = [config.DIFFICULTY_DELTA[d] for d in config.DIFFICULTIES]
    assert bands[0][0] == 0.0
    assert bands[-1][1] == 1.0
    for (lo1, hi1), (lo2, hi2) in zip(bands, bands[1:]):
        assert hi1 == lo2, "difficulty bands must be contiguous with no gaps or overlaps"


def test_every_evaluated_model_has_a_cost_entry():
    for key, model_id in config.MODELS.items():
        assert model_id in config.COST_PER_1M, f"{model_id} ({key}) missing from COST_PER_1M"
