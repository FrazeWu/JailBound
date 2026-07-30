from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchmark.safety_eval.datasets import RawExample
from benchmark.safety_eval.manifest import (
    AnnotationFailure,
    FolCandidate,
    annotate_raw_candidates,
    build_v2_controlled_manifests,
    select_controlled,
    select_fol_validation,
    write_v2_annotation_failures,
    write_v2_controlled_manifest,
    write_controlled_manifest,
    build_controlled_manifests,
)
from benchmark.safety_eval.schema import BenchmarkExample, EditableSpan, V2BenchmarkExample
from benchmark.safety_eval.semantic import MappingDecision
from benchmark.safety_eval.span_annotation import FrozenSpanAnnotation, SpanAnnotationError


APPROVED_SOURCES = (
    "advbench",
    "harmbench",
    "safetybench",
    "sg_bench",
    "jailbreakbench",
    "jailbound",
    "s_eval",
)


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


def _raw(source: str, index: int, *, prompt: str | None = None) -> RawExample:
    attack_text = prompt or f"{source} prompt {index}"
    return RawExample(
        source=source,
        source_row=index,
        intent=f"{source} intent {index}",
        attack_text=attack_text,
        target_text=None,
        source_risk_label=f"risk-{index % 2}",
        source_attack_label="direct_request",
        source_domain_label=f"domain-{index % 2}",
        language="en",
        preprocessing=("fixture",),
    )


def _mapping(raw: RawExample) -> MappingDecision:
    return MappingDecision(
        risk_category=f"risk_{raw.source_row % 3}",
        attack_type="direct_request",
        threat_domain=f"domain_{raw.source_row % 2}",
        routes={
            "risk_category": "fixture",
            "attack_type": "direct_request_control",
            "threat_domain": "fixture",
        },
        candidate_scores={
            "risk_category": {},
            "attack_type": {},
            "threat_domain": {},
        },
    )


class RecordingAnnotator:
    def __init__(self, *, failing_ids: frozenset[str] = frozenset()) -> None:
        self.calls: list[dict[str, object]] = []
        self.failing_ids = failing_ids

    def annotate(
        self,
        prompt: str,
        *,
        seed_intent: str,
        source_hints: dict[str, object],
    ) -> FrozenSpanAnnotation:
        self.calls.append(dict(source_hints))
        if source_hints["source_row_id"] in self.failing_ids:
            raise SpanAnnotationError("fixture invalid annotation")
        quote = prompt[: min(4, len(prompt))]
        return FrozenSpanAnnotation(
            spans=(
                EditableSpan(
                    start=0,
                    end=len(quote),
                    quote=quote,
                    role="harmful_payload",
                    confidence=0.95,
                    rationale="fixture",
                ),
            ),
            confidence=0.95,
            template_sha256="b" * 64,
            response_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            model="fixture-model",
            revision="fixture-revision",
        )


def _v2_rows(count: int, *, source: str = "synthetic") -> tuple[V2BenchmarkExample, ...]:
    raw = [_raw(source, index) for index in range(count)]
    rows, failures = annotate_raw_candidates(
        raw,
        annotator=RecordingAnnotator(),
        taxonomy_mapper=_mapping,
        source_file=f"{source}.jsonl",
        source_sha256="a" * 64,
        seed=20260725,
    )
    assert not failures
    return rows


def test_v2_candidates_are_annotated_before_selection_for_all_sources() -> None:
    raw = [_raw(source, index) for source in APPROVED_SOURCES for index in range(2)]
    annotator = RecordingAnnotator()
    mapped_after_annotation: list[str] = []

    def mapping(candidate: RawExample) -> MappingDecision:
        assert candidate.source_row_id in {
            call["source_row_id"] for call in annotator.calls
        }
        mapped_after_annotation.append(candidate.source_row_id)
        return _mapping(candidate)

    candidates, failures = annotate_raw_candidates(
        raw,
        annotator=annotator,
        taxonomy_mapper=mapping,
        source_file="fixture.jsonl",
        source_sha256="a" * 64,
        seed=20260725,
    )

    assert {call["source"] for call in annotator.calls} == set(APPROVED_SOURCES)
    assert len(mapped_after_annotation) == len(raw)
    assert all(row.schema_version == "reviewer_eval.v2" for row in candidates)
    assert not failures


def test_annotation_failures_are_audited_and_cannot_enter_selection(tmp_path) -> None:
    raw = [_raw("advbench", index) for index in range(4)]
    failed_id = raw[1].source_row_id
    candidates, failures = annotate_raw_candidates(
        raw,
        annotator=RecordingAnnotator(failing_ids=frozenset({failed_id})),
        taxonomy_mapper=_mapping,
        source_file="advbench.csv",
        source_sha256="a" * 64,
        seed=20260725,
    )

    assert {row.example_id for row in candidates}.isdisjoint(
        {failure.example_id for failure in failures}
    )
    assert [failure.failure_kind for failure in failures] == ["annotation"]
    write_v2_annotation_failures(tmp_path, failures)
    ledger = tmp_path / "manifests/v2/annotation_failures.jsonl"
    assert failed_id in ledger.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="need 4 unique eligible prompts, found 3"):
        build_v2_controlled_manifests(
            {"advbench": candidates},
            output_root=tmp_path,
            source_hashes={"advbench": "a" * 64},
            config_hash="c" * 64,
            seed=20260725,
            samples_per_source=4,
        )


@pytest.mark.parametrize(
    ("error", "expected_kind", "expected_code"),
    [
        (
            SpanAnnotationError("SECRET_PROMPT SECRET_RESPONSE request-id=abc"),
            "annotation",
            "annotation_error",
        ),
        (
            RuntimeError("SECRET_INTENT SECRET_RESPONSE request-id=xyz"),
            "transport",
            "transport_error",
        ),
    ],
)
def test_annotation_failure_ledger_never_persists_exception_content(
    tmp_path, error: Exception, expected_kind: str, expected_code: str
) -> None:
    class SecretFailingAnnotator:
        def annotate(self, *args, **kwargs):
            raise error

    _, failures = annotate_raw_candidates(
        [_raw("advbench", 0, prompt="SECRET_PROMPT")],
        annotator=SecretFailingAnnotator(),
        taxonomy_mapper=_mapping,
        source_file="advbench.csv",
        source_sha256="a" * 64,
        seed=20260725,
    )
    write_v2_annotation_failures(tmp_path, failures)
    persisted = (tmp_path / "manifests/v2/annotation_failures.jsonl").read_text(
        encoding="utf-8"
    )

    assert failures[0].failure_kind == expected_kind
    assert failures[0].failure_reason == expected_code
    assert "SECRET" not in str(failures[0].as_dict())
    assert "SECRET" not in persisted
    assert "request-id" not in persisted


def test_annotation_failure_record_rejects_content_bearing_reason() -> None:
    with pytest.raises(ValueError, match="stable failure code"):
        AnnotationFailure(
            schema_version="reviewer_eval.v2",
            example_id="advbench:000000:fixture",
            source="advbench",
            source_file="advbench.csv",
            source_row=0,
            source_sha256="a" * 64,
            prompt_sha256="b" * 64,
            intent_sha256="c" * 64,
            failure_kind="annotation",
            failure_reason="SpanAnnotationError: SECRET_RESPONSE",
        )


def test_v2_manifest_uses_isolated_schema_and_requires_byte_identical_rerun(
    tmp_path,
) -> None:
    records = _v2_rows(4)
    header = write_v2_controlled_manifest(
        tmp_path,
        "synthetic",
        records,
        source_file_sha256="a" * 64,
        config_hash="c" * 64,
    )
    manifest = tmp_path / "manifests/v2/controlled_synthetic.jsonl"
    exact_bytes = manifest.read_bytes()

    repeat = write_v2_controlled_manifest(
        tmp_path,
        "synthetic",
        records,
        source_file_sha256="a" * 64,
        config_hash="c" * 64,
    )
    assert repeat == header
    assert manifest.read_bytes() == exact_bytes
    assert header.schema_version == "reviewer_eval.v2"

    changed = list(records)
    changed[0] = changed[0].model_copy(
        update={
            "attack_text": "changed prompt",
            "prompt_sha256": hashlib.sha256(b"changed prompt").hexdigest(),
            "editable_spans": (
                EditableSpan(
                    start=0,
                    end=4,
                    quote="chan",
                    role="harmful_payload",
                    confidence=0.95,
                    rationale="fixture",
                ),
            ),
        }
    )
    with pytest.raises(ValueError, match="immutable v2 manifest differs"):
        write_v2_controlled_manifest(
            tmp_path,
            "synthetic",
            changed,
            source_file_sha256="a" * 64,
            config_hash="c" * 64,
        )


def test_v2_manifest_writer_rejects_legacy_records(tmp_path) -> None:
    with pytest.raises(ValueError, match="V2BenchmarkExample"):
        write_v2_controlled_manifest(
            tmp_path,
            "synthetic",
            [_example(0)],  # type: ignore[list-item]
            source_file_sha256="a" * 64,
            config_hash="c" * 64,
        )

    assert not (tmp_path / "manifests/v2/controlled_synthetic.jsonl").exists()


def test_v2_manifest_writer_validates_conflicting_header_before_jsonl_write(
    tmp_path,
) -> None:
    header = tmp_path / "manifests/v2/controlled_synthetic.header.json"
    header.parent.mkdir(parents=True)
    header.write_text('{"conflict":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="immutable v2 manifest differs"):
        write_v2_controlled_manifest(
            tmp_path,
            "synthetic",
            _v2_rows(1),
            source_file_sha256="a" * 64,
            config_hash="c" * 64,
        )

    assert not (tmp_path / "manifests/v2/controlled_synthetic.jsonl").exists()
    assert header.read_text(encoding="utf-8") == '{"conflict":true}\n'


def test_v2_manifest_builder_rejects_legacy_records_before_selection(tmp_path) -> None:
    with pytest.raises(ValueError, match="V2BenchmarkExample"):
        build_v2_controlled_manifests(
            {
                "valid": _v2_rows(1, source="valid"),
                "synthetic": [_example(0)],  # type: ignore[dict-item]
            },
            output_root=tmp_path,
            source_hashes={"valid": "b" * 64, "synthetic": "a" * 64},
            config_hash="c" * 64,
            seed=20260725,
            samples_per_source=1,
        )

    assert not list((tmp_path / "manifests/v2").glob("controlled_*.jsonl"))


def test_v2_manifest_builder_validates_all_sources_before_any_write(tmp_path) -> None:
    with pytest.raises(ValueError, match="requested source"):
        build_v2_controlled_manifests(
            {
                "first": _v2_rows(1, source="first"),
                "second": _v2_rows(1, source="different"),
            },
            output_root=tmp_path,
            source_hashes={"first": "a" * 64, "second": "b" * 64},
            config_hash="c" * 64,
            seed=20260725,
            samples_per_source=1,
        )

    manifests = tmp_path / "manifests/v2"
    assert not list(manifests.glob("controlled_*.jsonl"))
    assert not list(manifests.glob("controlled_*.header.json"))


def test_concurrent_v2_manifest_writers_cannot_overwrite_each_other(tmp_path) -> None:
    first = _v2_rows(3)
    second = tuple(
        row.model_copy(update={"selection_seed": 20260726}) for row in first
    )

    def write(records: tuple[V2BenchmarkExample, ...]) -> str:
        return write_v2_controlled_manifest(
            tmp_path,
            "synthetic",
            records,
            source_file_sha256="a" * 64,
            config_hash="c" * 64,
        ).manifest_hash

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write, rows) for rows in (first, second)]
    successes = [future.result() for future in futures if future.exception() is None]
    failures = [future.exception() for future in futures if future.exception() is not None]

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)


def test_controlled_selection_is_order_invariant_and_deduplicates() -> None:
    records = [_example(index) for index in range(60)] + [_example(99, prompt="synthetic prompt 1")]
    first, first_report = select_controlled(records, n=50, seed=20260725, coverage_dimensions=("risk_category", "attack_type"))
    second, _ = select_controlled(list(reversed(records)), n=50, seed=20260725, coverage_dimensions=("risk_category", "attack_type"))

    assert [row.example_id for row in first] == [row.example_id for row in second]
    assert len(first) == 50
    assert len({row.prompt_sha256 for row in first}) == 50
    assert first_report.duplicate_count == 1


def test_controlled_selection_rejects_duplicate_example_ids_before_quotas() -> None:
    first = _example(0)
    duplicate_id = _example(1).model_copy(update={"example_id": first.example_id})

    with pytest.raises(ValueError, match="example_id.*unique"):
        select_controlled(
            [first, duplicate_id, _example(2)],
            n=3,
            seed=20260725,
            coverage_dimensions=("risk_category", "attack_type"),
        )


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
