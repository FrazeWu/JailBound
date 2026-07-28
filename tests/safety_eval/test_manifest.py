from __future__ import annotations

import hashlib

import pytest

from benchmark.safety_eval.manifest import (
    FolCandidate,
    select_controlled,
    select_fol_validation,
    write_controlled_manifest,
    build_controlled_manifests,
)
from benchmark.safety_eval.schema import BenchmarkExample


def _example(index: int, *, prompt: str | None = None) -> BenchmarkExample:
    attack_text = prompt or f"synthetic prompt {index}"
    digest = hashlib.sha256(attack_text.encode()).hexdigest()
    return BenchmarkExample(
        example_id=f"synthetic:{index:03d}", source="synthetic", source_file="synthetic.jsonl",
        source_row=index, source_sha256="a" * 64, intent=f"intent {index}",
        attack_text=attack_text, target_text=None, source_risk_label=None,
        source_attack_label="direct_request", risk_category=f"risk_{index % 3}",
        threat_domain=f"domain_{index % 2}", attack_type=f"type_{index % 2}",
        language="en", selection_stratum=f"risk_{index % 3}|type_{index % 2}",
        selection_seed=20260725, prompt_sha256=digest, preprocessing=(),
    )


def test_controlled_selection_is_order_invariant_and_deduplicates() -> None:
    records = [_example(index) for index in range(60)] + [_example(99, prompt="synthetic prompt 1")]
    first, first_report = select_controlled(records, n=50, seed=20260725, coverage_dimensions=("risk_category", "attack_type"))
    second, _ = select_controlled(list(reversed(records)), n=50, seed=20260725, coverage_dimensions=("risk_category", "attack_type"))

    assert [row.example_id for row in first] == [row.example_id for row in second]
    assert len(first) == 50
    assert len({row.prompt_sha256 for row in first}) == 50
    assert first_report.duplicate_count == 1


def test_fol_split_uses_prime_reported_groups_and_is_disjoint() -> None:
    rows = [FolCandidate(sample_id=f"sample:{index:03d}", source="synthetic", fol=float(index),
                         risk_category=f"risk_{index % 2}", initial_label=bool(index % 2),
                         attack_loss=float(index % 2), token_length=20 + index % 2,
                         perplexity=5.0 + index % 2) for index in range(45)]
    split = select_fol_validation(rows, validation_n=17, low_n=7, middle_n=3, high_n=7)

    assert [len(split.low), len(split.middle), len(split.high)] == [7, 3, 7]
    assert not ({row.sample_id for row in split.low} & {row.sample_id for row in split.high})
    assert not ({row.sample_id for row in split.middle} & {row.sample_id for row in split.high})


def test_fol_split_requires_matched_initial_labels() -> None:
    rows = [
        FolCandidate(
            sample_id=f"sample:{index:03d}", source="synthetic", fol=float(index),
            risk_category="risk", initial_label=index < 18, attack_loss=1.0,
            token_length=20, perplexity=5.0,
        )
        for index in range(45)
    ]

    split = select_fol_validation(rows, validation_n=17, low_n=7, middle_n=3, high_n=7)

    assert split.status == "inconclusive"


def test_fol_split_uses_first_predeclared_relaxed_caliper_when_strict_matching_is_insufficient() -> None:
    rows = [
        FolCandidate(
            sample_id=f"sample:{index:03d}", source="synthetic", fol=float(index),
            risk_category="risk", initial_label=True,
            attack_loss=0.0 if index < 18 else (3.0 if index < 27 else 0.7),
            token_length=20, perplexity=5.0,
        )
        for index in range(45)
    ]

    split = select_fol_validation(rows, validation_n=17, low_n=7, middle_n=3, high_n=7)

    assert split.status == "ready"
    assert split.matching_caliper == pytest.approx(0.75)
    assert len(split.matching_distances) == 7
    assert all(0.5 < distance <= 0.75 for distance in split.matching_distances)


def test_controlled_manifest_is_content_addressed_and_immutable(tmp_path) -> None:
    records, _ = select_controlled([_example(index) for index in range(50)], n=50, seed=20260725, coverage_dimensions=("risk_category", "attack_type"))
    header = write_controlled_manifest(tmp_path, "synthetic", records, source_file_sha256="a" * 64, config_hash="b" * 64)
    repeat = write_controlled_manifest(tmp_path, "synthetic", records, source_file_sha256="a" * 64, config_hash="b" * 64)

    assert repeat == header
    assert header.record_count == 50
    changed = list(records)
    changed[0] = changed[0].model_copy(update={"attack_text": "different", "prompt_sha256": "c" * 64})
    with pytest.raises(ValueError, match="immutable manifest differs"):
        write_controlled_manifest(tmp_path, "synthetic", changed, source_file_sha256="a" * 64, config_hash="b" * 64)


def test_build_controlled_manifests_selects_each_audited_source(tmp_path) -> None:
    records = [_example(index) for index in range(50)]
    headers = build_controlled_manifests(
        {"synthetic": records},
        output_root=tmp_path,
        source_hashes={"synthetic": "a" * 64},
        config_hash="b" * 64,
        seed=20260725,
        samples_per_source=17,
    )
    assert headers["synthetic"].record_count == 17
