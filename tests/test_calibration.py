"""Tests for the pure-numeric parts of consistencybench.calibration: the CCS
(consistency-calibration ECE) computation itself."""
from consistencybench.calibration import compute_ccs


def test_perfectly_calibrated_predictions_give_zero_ccs():
    # Every bin's mean predicted probability equals its mean observed rate.
    p_pred = [0.1] * 10 + [0.9] * 10
    y_true = [0] * 9 + [1] * 1 + [1] * 9 + [0] * 1  # 10% and 90% observed rates
    ccs, bins = compute_ccs(p_pred, y_true, n_bins=10)
    assert ccs < 0.05  # near-zero calibration error


def test_badly_calibrated_predictions_give_high_ccs():
    # Model claims high confidence of inconsistency (0.9) but is actually
    # always consistent (observed rate 0).
    p_pred = [0.9] * 20
    y_true = [0] * 20
    ccs, bins = compute_ccs(p_pred, y_true, n_bins=10)
    assert ccs > 0.5


def test_compute_ccs_handles_empty_input():
    ccs, bins = compute_ccs([], [])
    assert ccs == 0.0
    assert bins == []
