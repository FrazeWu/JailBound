from __future__ import annotations

import pytest

from benchmark.reviewer_eval.analysis import (
    FolClaimEvidence,
    FolFlipPrediction,
    decide_fol_claim,
    grouped_flip_prediction_comparison,
)
from benchmark.reviewer_eval.fol_boundary import (
    MarginCalibrationRow,
    fit_margin_calibration,
    locate_margin_crossing,
    estimate_behavior_distance,
)


def test_margin_calibration_excludes_validation_ids_and_interpolates_threshold() -> None:
    rows = [
        MarginCalibrationRow("train:0", -2.0, False),
        MarginCalibrationRow("train:1", -1.0, False),
        MarginCalibrationRow("validation:0", 0.0, True),
        MarginCalibrationRow("train:2", 1.0, True),
        MarginCalibrationRow("train:3", 2.0, True),
    ]

    calibration = fit_margin_calibration(rows, excluded_ids={"validation:0"})

    assert calibration.training_ids == ("train:0", "train:1", "train:2", "train:3")
    assert calibration.threshold == pytest.approx(0.0)
    assert calibration.brier_score == pytest.approx(0.0)


def test_margin_crossing_returns_bisection_interval() -> None:
    crossing = locate_margin_crossing(
        lambda radius: radius - 0.3,
        threshold=0.0,
        lower=0.0,
        upper=1.0,
        iterations=12,
    )

    assert crossing.lower < 0.3 < crossing.upper
    assert crossing.estimate == pytest.approx(0.3, abs=1 / 2**12)
    assert crossing.upper - crossing.lower == pytest.approx(1 / 2**12)


def test_claim_ladder_requires_h1_h2_h3_and_quality_gates() -> None:
    evidence = FolClaimEvidence(
        h1=True,
        h2=True,
        h3=True,
        h4=False,
        secondary_same_direction=True,
        valid_paths=5,
        usable_fraction=0.9,
        band_acceptance_difference=0.1,
        h1_interval_width=0.2,
        h2_interval_width=0.3,
    )

    assert decide_fol_claim(evidence) == "boundary_proxy_support"
    assert decide_fol_claim(FolClaimEvidence(**{**evidence.__dict__, "h2": False})) == "local_sensitivity_support_only"
    assert decide_fol_claim(FolClaimEvidence(**{**evidence.__dict__, "valid_paths": 4})) == "inconclusive"


def test_behavior_distance_uses_isotonic_crossing_or_right_censoring() -> None:
    crossed = estimate_behavior_distance({0.1: 0.0, 0.2: 0.25, 0.4: 0.75, 0.8: 1.0})
    censored = estimate_behavior_distance({0.1: 0.0, 0.2: 0.1, 0.4: 0.2, 0.8: 0.4})

    assert crossed.estimate == pytest.approx(0.3)
    assert crossed.right_censored is False
    assert censored.estimate is None
    assert censored.lower == pytest.approx(0.8)
    assert censored.right_censored is True


def test_grouped_flip_comparison_evaluates_fol_on_held_out_prompt_groups() -> None:
    rows = [
        FolFlipPrediction(
            prompt_id=f"prompt:{prompt_index}",
            flipped=fol > 0.5,
            fol=fol,
            controls=(0.0, 1.0),
        )
        for prompt_index in range(12)
        for fol in (0.0, 0.0, 1.0, 1.0)
    ]

    comparison = grouped_flip_prediction_comparison(rows, folds=4, seed=7)

    assert comparison.held_out_rows == len(rows)
    assert comparison.controls_auroc == pytest.approx(0.5)
    assert comparison.fol_auroc == pytest.approx(1.0)
    assert comparison.delta_auroc == pytest.approx(0.5)
    assert comparison.delta_auprc > 0.0
