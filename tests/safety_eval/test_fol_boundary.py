from __future__ import annotations

import inspect

import pytest
import torch

from benchmark.safety_eval.fol_boundary import (
    FolPerturbationOutcome,
    FolValue,
    InterpolationPoint,
    LabeledEditableState,
    behavior_flip_rate,
    fit_right_censored_cox,
    exact_permutation_mean_difference,
    has_minimum_valid_paths,
    interpolate_joint_states,
    random_joint_direction,
    select_primary_behavioral_radius,
    select_nearest_opposite_label_pairs,
    select_base_radius,
    split_fol_bands,
    summarize_interpolation_peaks,
    summarize_fol_bfr,
    validate_interpolation_path,
)


def test_random_joint_direction_has_unit_joint_norm_and_only_shape_generator_inputs() -> None:
    generator = torch.Generator().manual_seed(7)

    dz, du = random_joint_direction((1, 2, 3), (1, 4, 3), generator)

    assert torch.linalg.vector_norm(torch.cat((dz.reshape(-1), du.reshape(-1)))).item() == pytest.approx(1.0)
    assert tuple(inspect.signature(random_joint_direction).parameters) == ("z_shape", "u_shape", "generator")
    assert dz.device.type == du.device.type == "cpu"


def test_select_base_radius_uses_only_boolean_semantic_acceptance() -> None:
    acceptance = {
        0.025: [True] * 10,
        0.05: [True] * 9 + [False],
        0.1: [True] * 8 + [False] * 2,
        0.2: [True] * 6 + [False] * 4,
    }

    assert select_base_radius(acceptance, minimum_rate=0.8) == pytest.approx(0.1)
    assert select_base_radius({0.1: [False] * 3}, minimum_rate=0.8) is None
    with pytest.raises(TypeError, match="booleans"):
        select_base_radius({0.1: [1, 0]}, minimum_rate=0.8)


def test_select_base_radius_requires_each_source_to_meet_its_acceptance_floor() -> None:
    acceptance = {0.1: [True] * 16 + [False] * 4}
    by_source = {
        "source_a": {0.1: [True] * 9 + [False]},
        "source_b": {0.1: [True] * 7 + [False] * 3},
    }

    assert select_base_radius(
        acceptance,
        minimum_rate=0.8,
        source_acceptance=by_source,
        minimum_source_rate=0.75,
    ) is None


def test_select_primary_behavioral_radius_requires_semantic_acceptance_and_targets_local_flips() -> None:
    acceptance = {
        0.1: [True] * 10,
        0.2: [True] * 10,
        0.4: [True] * 7 + [False] * 3,
        0.8: [True] * 10,
    }
    primary_bfr = {0.1: 0.05, 0.2: 0.30, 0.4: 0.40, 0.8: 0.70}

    assert select_primary_behavioral_radius(
        acceptance,
        primary_bfr,
        target_lower=0.2,
        target_upper=0.5,
        minimum_semantic_rate=0.8,
    ) == pytest.approx(0.2)
    assert select_primary_behavioral_radius(
        acceptance,
        {radius: 0.05 for radius in acceptance},
        target_lower=0.2,
        target_upper=0.5,
        minimum_semantic_rate=0.8,
    ) is None


def test_split_fol_bands_is_deterministic_and_enforces_disjoint_7_3_7_ids() -> None:
    values = [FolValue(sample_id=f"sample:{index:02d}", fol=float(index)) for index in range(45)]

    split = split_fol_bands(list(reversed(values)))

    assert [len(split.low), len(split.middle), len(split.high)] == [7, 3, 7]
    assert set(split.low).isdisjoint(split.middle)
    assert set(split.low).isdisjoint(split.high)
    assert set(split.middle).isdisjoint(split.high)
    assert split.low == tuple(f"sample:{index:02d}" for index in range(7))
    assert split.high == tuple(f"sample:{index:02d}" for index in range(38, 45))
    with pytest.raises(ValueError, match="at least 17"):
        split_fol_bands(values[:16])
    with pytest.raises(ValueError, match="duplicate"):
        split_fol_bands(values + [FolValue(sample_id="sample:00", fol=99.0)])


def test_behavior_flip_rate_censors_insufficient_accepted_directions() -> None:
    censored = behavior_flip_rate([True, False, True], minimum_accepted=4)
    measured = behavior_flip_rate([True, False, True, False], minimum_accepted=4)

    assert censored.rate is None
    assert censored.acceptance_failure is True
    assert measured.rate == pytest.approx(0.5)
    assert measured.acceptance_failure is False


def test_exact_permutation_mean_difference_uses_two_sided_label_permutations() -> None:
    difference, pvalue = exact_permutation_mean_difference([0.0, 0.0], [1.0, 1.0])

    assert difference == pytest.approx(1.0)
    assert pvalue == pytest.approx(2 / 6)


def test_right_censored_cox_recovers_positive_hazard_for_higher_covariates() -> None:
    estimate = fit_right_censored_cox(
        covariates=[-1.0, -0.5, 0.5, 1.0],
        times=[4.0, 3.0, 2.0, 1.0],
        events=[False, True, True, True],
    )

    assert estimate.event_count == 3
    assert estimate.log_hazard_ratio > 0.0
    assert estimate.hazard_ratio > 1.0
    assert 0.0 <= estimate.pvalue <= 1.0


def test_fol_bfr_summary_excludes_sparse_prompt_direction_sets() -> None:
    rows = [
        FolPerturbationOutcome("source", "sample:one", "low", 0.1, True, True, False, False, False),
        FolPerturbationOutcome("source", "sample:one", "low", 0.1, True, False, False, False, False),
        FolPerturbationOutcome("source", "sample:one", "low", 0.1, True, True, False, False, False),
        FolPerturbationOutcome("source", "sample:one", "low", 0.1, True, False, False, False, False),
        FolPerturbationOutcome("source", "sample:two", "low", 0.1, True, True, False, False, False),
        FolPerturbationOutcome("source", "sample:two", "low", 0.1, True, False, False, False, False),
        FolPerturbationOutcome("source", "sample:two", "low", 0.1, True, True, False, False, False),
    ]

    summaries = summarize_fol_bfr(rows, minimum_accepted=4)

    assert len(summaries) == 2
    primary = next(row for row in summaries if row.judge_key == "primary")
    secondary = next(row for row in summaries if row.judge_key == "secondary")
    assert (primary.source, primary.band, primary.radius) == ("source", "low", 0.1)
    assert (primary.prompt_count, primary.eligible_prompt_count, primary.sparse_prompt_count) == (2, 1, 1)
    assert primary.accepted_direction_count == 7
    assert primary.mean_bfr == pytest.approx(0.5)
    assert secondary.mean_bfr == pytest.approx(0.0)


def test_fol_bfr_summary_omits_secondary_when_only_primary_labels_exist() -> None:
    rows = [
        FolPerturbationOutcome("source", "sample:one", "high", 0.2, True, True, False, None, None),
        FolPerturbationOutcome("source", "sample:one", "high", 0.2, True, False, False, None, None),
        FolPerturbationOutcome("source", "sample:one", "high", 0.2, True, True, False, None, None),
        FolPerturbationOutcome("source", "sample:one", "high", 0.2, True, False, False, None, None),
    ]

    summaries = summarize_fol_bfr(rows, minimum_accepted=4)

    assert [summary.judge_key for summary in summaries] == ["primary"]
    assert summaries[0].mean_bfr == pytest.approx(0.5)


def test_interpolate_joint_states_has_21_inclusive_linear_points() -> None:
    safe_z = torch.zeros((1, 1, 2))
    safe_u = torch.zeros((1, 2, 2))
    unsafe_z = torch.full((1, 1, 2), 2.0)
    unsafe_u = torch.full((1, 2, 2), 4.0)

    points = interpolate_joint_states(safe_z, safe_u, unsafe_z, unsafe_u)

    assert len(points) == 21
    assert torch.equal(points[0][0], safe_z)
    assert torch.equal(points[0][1], safe_u)
    assert torch.equal(points[-1][0], unsafe_z)
    assert torch.equal(points[-1][1], unsafe_u)
    assert torch.equal(points[10][0], torch.full((1, 1, 2), 1.0))
    assert torch.equal(points[10][1], torch.full((1, 2, 2), 2.0))


def test_interpolation_validity_requires_endpoints_17_points_and_5_paths() -> None:
    accepted = [True] * 16 + [False] * 4 + [True]
    invalid_endpoint = [False] + [True] * 20

    assert validate_interpolation_path(accepted, minimum_valid=17) is True
    assert validate_interpolation_path([True] * 15 + [False] * 5 + [True], minimum_valid=17) is False
    assert validate_interpolation_path(invalid_endpoint, minimum_valid=17) is False
    assert has_minimum_valid_paths([True] * 5 + [False] * 2) is True
    assert has_minimum_valid_paths([True] * 4 + [False] * 3) is False
    with pytest.raises(ValueError, match="exactly 21"):
        validate_interpolation_path([True] * 20)


def test_interpolation_peak_summary_compares_fol_to_random_and_curvature_locations() -> None:
    paths = []
    for path_index in range(5):
        points = []
        for point_index in range(21):
            points.append(InterpolationPoint(
                path_id=f"path:{path_index}",
                point_index=point_index,
                semantic_accepted=True,
                unsafe_label=point_index >= 10,
                fol=10.0 if point_index == 10 else 0.0,
                curvature=10.0 if point_index == 1 else 0.0,
            ))
        paths.append(tuple(points))

    summary = summarize_interpolation_peaks(paths, minimum_valid=17, minimum_paths=5, permutations=10_000, seed=7)

    assert summary.valid_path_count == 5
    assert summary.crossing_path_count == 5
    assert summary.fol_mean_distance == pytest.approx(0.5)
    assert summary.curvature_mean_distance == pytest.approx(8.5)
    assert summary.random_mean_distance is not None
    assert summary.fol_mean_distance < summary.random_mean_distance
    assert summary.fol_vs_random_pvalue is not None and summary.fol_vs_random_pvalue < 0.05
    assert summary.fol_vs_curvature_pvalue is not None and summary.fol_vs_curvature_pvalue < 0.05


def test_nearest_opposite_label_pairs_use_only_labels_and_editable_state_distance() -> None:
    candidates = [
        LabeledEditableState("sample:a", "safe:far", False, torch.tensor([0.0, 0.0])),
        LabeledEditableState("sample:a", "safe:near", False, torch.tensor([4.8, 0.0])),
        LabeledEditableState("sample:a", "unsafe", True, torch.tensor([5.0, 0.0])),
        LabeledEditableState("sample:b", "only-safe", False, torch.tensor([0.0, 0.0])),
    ]

    pairs = select_nearest_opposite_label_pairs(candidates)

    assert len(pairs) == 1
    assert pairs[0].sample_id == "sample:a"
    assert pairs[0].safe_candidate_id == "safe:near"
    assert pairs[0].unsafe_candidate_id == "unsafe"
    assert pairs[0].distance == pytest.approx(0.2)
