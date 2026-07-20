"""Tests for consistencybench.profiles using synthetic scored-result fixtures
(no network / no real experiment data required)."""
from consistencybench import config
from consistencybench.profiles import build_results_dataframe, compute_cp, profile_distance_matrix


def _fake_result(model, family, consistent, domain="general", difficulty="easy"):
    return {"model": model, "family": family, "domain": domain, "difficulty": difficulty,
            "delta": 0.2, "consistent": consistent, "score_method": "rbs"}


def test_compute_cp_is_100_when_always_inconsistent():
    fam = config.TRANSFORMATION_FAMILIES[0]
    results = [_fake_result("m1", fam, False) for _ in range(5)]
    df = build_results_dataframe(results)
    cp = compute_cp("m1", df)
    idx = config.TRANSFORMATION_FAMILIES.index(fam)
    assert cp[idx] == 100.0


def test_compute_cp_is_0_when_always_consistent():
    fam = config.TRANSFORMATION_FAMILIES[0]
    results = [_fake_result("m1", fam, True) for _ in range(5)]
    df = build_results_dataframe(results)
    cp = compute_cp("m1", df)
    idx = config.TRANSFORMATION_FAMILIES.index(fam)
    assert cp[idx] == 0.0


def test_identical_profiles_have_zero_distance():
    fam = config.TRANSFORMATION_FAMILIES[0]
    results = [_fake_result("m1", fam, True), _fake_result("m2", fam, True)]
    df = build_results_dataframe(results)
    cp_vectors = {"m1": compute_cp("m1", df), "m2": compute_cp("m2", df)}
    dist = profile_distance_matrix(cp_vectors)
    assert dist.loc["m1", "m2"] == 0.0


def test_different_profiles_have_positive_distance():
    fam = config.TRANSFORMATION_FAMILIES[0]
    results = [_fake_result("m1", fam, True), _fake_result("m2", fam, False)]
    df = build_results_dataframe(results)
    cp_vectors = {"m1": compute_cp("m1", df), "m2": compute_cp("m2", df)}
    dist = profile_distance_matrix(cp_vectors)
    assert dist.loc["m1", "m2"] > 0.0
