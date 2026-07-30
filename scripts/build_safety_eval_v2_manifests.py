"""Build immutable reviewer_eval.v2 manifests from annotated raw candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, NamedTuple

from benchmark.safety_eval.config import AnnotationConfig, load_v2_config
from benchmark.safety_eval.datasets import RawExample, load_source_with_report
from benchmark.safety_eval.io import atomic_write_json, canonical_hash
from benchmark.safety_eval.manifest import (
    AnnotationFailure,
    annotate_raw_candidates,
    build_v2_controlled_manifests,
    write_v2_annotation_failures,
    write_v2_build_report,
)
from benchmark.safety_eval.runtime import LockedRuntime, lock_runtime_config
from benchmark.safety_eval.semantic import (
    MappingDecision,
    QwenHiddenMeanEncoder,
    load_taxonomy_mapping,
    map_raw_example,
)
from benchmark.safety_eval.span_annotation import SpanAnnotator


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from calibrate_span_annotation_confidence import OpenAIAnnotationTransport  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
APPROVED_ARTIFACT_VERSION = "span_annotation_confidence.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BuildInputSnapshots(NamedTuple):
    paths: Mapping[str, Path]
    hashes: Mapping[str, str]


@contextmanager
def snapshot_build_inputs(
    paths: Mapping[str, Path],
) -> Iterator[BuildInputSnapshots]:
    """Yield read-only temp files whose hashes come from the exact same bytes."""
    with tempfile.TemporaryDirectory(prefix="safety-eval-v2-inputs-") as directory:
        snapshot_root = Path(directory)
        snapshot_paths: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        for index, (name, original_path) in enumerate(sorted(paths.items())):
            original = Path(original_path)
            payload = original.read_bytes()
            target_dir = snapshot_root / f"{index:03d}"
            target_dir.mkdir()
            target = target_dir / original.name
            target.write_bytes(payload)
            target.chmod(0o400)
            snapshot_paths[name] = target
            hashes[name] = hashlib.sha256(payload).hexdigest()
        yield BuildInputSnapshots(paths=snapshot_paths, hashes=hashes)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=int, default=200)
    parser.add_argument("--sources", action="append")
    return parser.parse_args(argv)


def _rooted(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _config_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name == "benchmark" and resolved.parent.parent.name == "configs":
        return resolved.parents[2]
    return ROOT


def _selected_sources(
    configured: Sequence[str], requested: Sequence[str] | None
) -> tuple[str, ...]:
    if requested is None:
        return tuple(configured)
    if not requested:
        raise ValueError("at least one configured source is required")
    duplicates = sorted(
        source for source, count in Counter(requested).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate configured source selector: {', '.join(duplicates)}")
    unknown = sorted(set(requested) - set(configured))
    if unknown:
        raise ValueError(f"unknown configured source: {', '.join(unknown)}")
    return tuple(source for source in configured if source in requested)


def _reject_v1_manifests(output_root: Path) -> None:
    legacy_root = output_root / "manifests"
    legacy = sorted(
        path
        for pattern in ("controlled_*.jsonl", "controlled_*.header.json")
        for path in legacy_root.rglob(pattern)
        if path.parent != legacy_root / "v2"
    )
    if legacy:
        raise ValueError(f"v2 output root contains a v1 manifest: {legacy[0]}")


def _load_confidence_threshold(
    path: Path,
    *,
    annotation: AnnotationConfig,
    configured_sources: Sequence[str],
) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen annotation confidence artifact is unreadable") from error
    if not isinstance(payload, dict):
        raise ValueError("frozen annotation confidence artifact must be an object")
    if payload.get("artifact_version") != APPROVED_ARTIFACT_VERSION:
        raise ValueError("frozen annotation confidence artifact version differs")
    threshold = payload.get("threshold")
    if type(threshold) not in (int, float) or not 0.0 <= threshold <= 1.0:
        raise ValueError("frozen annotation confidence threshold must be in [0, 1]")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not all(
        isinstance(item, str) for item in sources
    ):
        raise ValueError("frozen annotation confidence sources must be strings")
    if len(sources) != len(set(sources)) or set(sources) != set(configured_sources):
        raise ValueError(
            "frozen annotation confidence sources differ from configured sources"
        )
    provenance = payload.get("annotation_provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError("frozen annotation provenance is required")
    template_sha256 = hashlib.sha256(
        Path(annotation.template_path).read_bytes()
    ).hexdigest()
    expected = (annotation.model, annotation.revision, template_sha256)
    for entry in provenance:
        if not isinstance(entry, dict):
            raise ValueError("frozen annotation provenance entries must be objects")
        identity = (
            entry.get("model"),
            entry.get("revision"),
            entry.get("template_sha256"),
        )
        response_sha256 = entry.get("response_sha256")
        if (
            identity != expected
            or not isinstance(response_sha256, str)
            or _SHA256.fullmatch(response_sha256) is None
        ):
            raise ValueError("frozen annotation provenance differs from locked config")
    return float(threshold)


def resolve_annotation_transport(
    annotation: AnnotationConfig,
    *,
    seed: int,
    api_key: str | None = None,
) -> OpenAIAnnotationTransport:
    """Resolve the same revision-pinned OpenAI-compatible transport as calibration."""
    return OpenAIAnnotationTransport(
        model=annotation.model,
        revision=annotation.revision,
        endpoint=annotation.endpoint,
        api_key=(
            api_key
            if api_key is not None
            else os.environ.get("OPENAI_API_KEY", "EMPTY")
        ),
        seed=seed,
    )


def _sample_key(seed: int, row: RawExample) -> str:
    return hashlib.sha256(f"{seed}|{row.source_row_id}".encode("utf-8")).hexdigest()


def _unique_prompt_count(rows: Sequence[Any]) -> int:
    return len({row.prompt_sha256 for row in rows})


def _lock_v2_runtime_config(
    config: Any,
    *,
    output_root: Path,
    source_hashes: dict[str, str],
    build_input_hashes: dict[str, str],
) -> LockedRuntime:
    """Create the runtime lock once and validate exact reruns without clobbering."""
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".v2_manifest_builder.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            config_payload = config.model_dump(mode="json")
            config_hash = canonical_hash(config_payload)
            ordered_hashes = dict(sorted(source_hashes.items()))
            ordered_build_input_hashes = dict(sorted(build_input_hashes.items()))
            run_id = (
                "run:"
                + canonical_hash(
                    {
                        "config_hash": config_hash,
                        "sources": ordered_hashes,
                        "build_inputs": ordered_build_input_hashes,
                    }
                )[:20]
            )
            config_lock = output_root / config.run.locked_config_name
            run_manifest = output_root / "run_manifest.json"
            if config_lock.exists() or run_manifest.exists():
                if not config_lock.is_file() or not run_manifest.is_file():
                    raise ValueError("incomplete immutable v2 runtime lock")
                try:
                    existing_config = json.loads(
                        config_lock.read_text(encoding="utf-8")
                    )
                    existing_manifest = json.loads(
                        run_manifest.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, json.JSONDecodeError) as error:
                    raise ValueError("immutable v2 runtime lock is unreadable") from error
                expected_manifest_fields = {
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "source_hashes": ordered_hashes,
                    "build_input_hashes": ordered_build_input_hashes,
                }
                if existing_config != config_payload or not isinstance(
                    existing_manifest, dict
                ) or any(
                    existing_manifest.get(key) != value
                    for key, value in expected_manifest_fields.items()
                ):
                    raise ValueError(
                        "immutable v2 runtime lock differs from requested build"
                    )
                return LockedRuntime(config, config_hash, run_id)
            locked = lock_runtime_config(
                config,
                output_root=output_root,
                source_hashes=source_hashes,
            )
            manifest_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
            manifest_payload["run_id"] = run_id
            manifest_payload["build_input_hashes"] = ordered_build_input_hashes
            atomic_write_json(run_manifest, manifest_payload)
            return LockedRuntime(locked.config, locked.config_hash, run_id)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def audit_and_require_sufficient_candidates(
    *,
    output_root: Path,
    candidates: dict[str, Sequence[Any]],
    failures: Sequence[AnnotationFailure],
    sources: Sequence[str],
    required: int,
    report: dict[str, object] | None = None,
) -> None:
    """Persist all annotation failures before enforcing controlled-sample quotas."""
    write_v2_annotation_failures(output_root, failures)
    eligible_counts = {
        source: _unique_prompt_count(candidates[source]) for source in sources
    }
    failures_by_source = Counter(failure.source for failure in failures)
    failure_counts_by_source_and_kind = {
        source: dict(
            sorted(
                Counter(
                    failure.failure_kind.value
                    for failure in failures
                    if failure.source == source
                ).items()
            )
        )
        for source in sources
    }
    insufficient = {
        source: (eligible_counts[source], failures_by_source[source])
        for source in sources
        if eligible_counts[source] < required
    }
    if insufficient:
        failure_report = dict(report or {})
        failure_report.update(
            {
                "schema_version": "reviewer_eval.v2",
                "status": "insufficient_candidates",
                "required_counts_by_source": {
                    source: required for source in sorted(sources)
                },
                "eligible_counts_by_source": dict(sorted(eligible_counts.items())),
                "failure_counts_by_source_and_kind": dict(
                    sorted(failure_counts_by_source_and_kind.items())
                ),
            }
        )
        write_v2_build_report(output_root, failure_report)
        detail = "; ".join(
            f"{source}: eligible={eligible}, failures={failure_count}, required={required}"
            for source, (eligible, failure_count) in insufficient.items()
        )
        raise ValueError(
            f"insufficient successfully annotated unique candidates: {detail}"
        )


def build_manifests(
    config_path: str | Path,
    *,
    candidate_pool: int,
    sources: Sequence[str] | None,
) -> dict[str, object]:
    """Run v2 manifest construction; model calls occur only through the transport."""
    if type(candidate_pool) is not int or candidate_pool < 1:
        raise ValueError("candidate_pool must be a positive integer")
    original_config_path = Path(config_path)
    root = _config_root(original_config_path)
    with snapshot_build_inputs(
        {"config": original_config_path}
    ) as config_snapshot:
        config = load_v2_config(config_snapshot.paths["config"])
        selected_sources = _selected_sources(config.data.sources, sources)
        output_root = _rooted(root, config.run.output_root)
        _reject_v1_manifests(output_root)

        taxonomy_mapping_path = (
            root / "configs/benchmark/safety_eval_taxonomy_map.yaml"
        )
        remaining_input_paths = {
            "confidence_artifact": _rooted(
                root, config.annotation.confidence_artifact
            ),
            "annotation_template": _rooted(
                root, config.annotation.template_path
            ),
            "taxonomy_mapping": taxonomy_mapping_path,
            **{
                f"source:{source}": _rooted(root, config.data.paths[source])
                for source in selected_sources
            },
        }
        if "harmbench" in selected_sources:
            remaining_input_paths["harmbench_targets"] = _rooted(
                root, config.data.harmbench_targets_path
            )

        with snapshot_build_inputs(remaining_input_paths) as inputs:
            annotation_payload = config.annotation.model_dump(mode="json")
            annotation_payload["template_path"] = inputs.paths[
                "annotation_template"
            ]
            annotation_payload["confidence_artifact"] = inputs.paths[
                "confidence_artifact"
            ]
            annotation_config = AnnotationConfig.model_validate(
                annotation_payload
            )
            threshold = _load_confidence_threshold(
                annotation_config.confidence_artifact,
                annotation=annotation_config,
                configured_sources=config.data.sources,
            )
            source_hashes = {
                source: inputs.hashes[f"source:{source}"]
                for source in selected_sources
            }
            build_input_hashes = {
                "config": config_snapshot.hashes["config"],
                **{
                    name: inputs.hashes[name]
                    for name in (
                        "confidence_artifact",
                        "annotation_template",
                        "taxonomy_mapping",
                    )
                },
            }
            if "harmbench" in selected_sources:
                build_input_hashes["harmbench_targets"] = inputs.hashes[
                    "harmbench_targets"
                ]
            locked = _lock_v2_runtime_config(
                config,
                output_root=output_root,
                source_hashes=source_hashes,
                build_input_hashes=build_input_hashes,
            )
            transport = resolve_annotation_transport(
                annotation_config, seed=config.run.seed
            )
            annotator = SpanAnnotator(
                annotation_config,
                transport,
                confidence_threshold=threshold,
            )

            mapping = load_taxonomy_mapping(inputs.paths["taxonomy_mapping"])
            labels = list(mapping["risk_categories"]) + list(
                mapping["threat_domains"]
            )
            descriptions = [
                mapping["risk_categories"].get(
                    label, mapping["threat_domains"].get(label)
                )["description"]
                for label in labels
            ]
            encoder = QwenHiddenMeanEncoder(
                config.models.semantic_encoder.local_path
            )
            vectors = encoder.encode(descriptions)
            embeddings = dict(zip(labels, vectors, strict=True))

            candidates: dict[str, tuple[Any, ...]] = {}
            all_failures: list[AnnotationFailure] = []
            reports: dict[str, dict[str, object]] = {}
            harmbench_targets = inputs.paths.get("harmbench_targets")
            for source in selected_sources:
                raw, load_report = load_source_with_report(
                    source,
                    inputs.paths[f"source:{source}"],
                    harmbench_targets if source == "harmbench" else None,
                )
                chosen = tuple(
                    sorted(
                        raw,
                        key=lambda row: _sample_key(config.run.seed, row),
                    )[:candidate_pool]
                )
                mapping_audit: dict[str, object] = {}

                def taxonomy_mapper(row: RawExample) -> MappingDecision:
                    decision = map_raw_example(
                        row, mapping, embeddings, encoder
                    )
                    mapping_audit[row.source_row_id] = decision.audit_entry
                    return decision

                annotated, failures = annotate_raw_candidates(
                    chosen,
                    annotator=annotator,
                    taxonomy_mapper=taxonomy_mapper,
                    source_file=str(config.data.paths[source]),
                    source_sha256=source_hashes[source],
                    seed=config.run.seed,
                )
                unique_eligible = _unique_prompt_count(annotated)
                failures_by_kind = Counter(
                    failure.failure_kind.value for failure in failures
                )
                reports[source] = {
                    "raw_count": load_report.raw_count,
                    "source_eligible_count": load_report.eligible_count,
                    "exclusions": dict(sorted(load_report.exclusions.items())),
                    "candidate_pool": len(chosen),
                    "annotated_count": len(annotated),
                    "annotation_failure_count": len(failures),
                    "unique_eligible_count": unique_eligible,
                    "failure_counts": dict(sorted(failures_by_kind.items())),
                    "mapping_audit": dict(sorted(mapping_audit.items())),
                }
                candidates[source] = annotated
                all_failures.extend(failures)

            report_identity = {
                "schema_version": "reviewer_eval.v2",
                "run_id": locked.run_id,
                "config_hash": locked.config_hash,
                "source_hashes": dict(sorted(source_hashes.items())),
                "build_input_hashes": dict(sorted(build_input_hashes.items())),
                "encoder_revision": encoder.resolved_revision,
                "sources": reports,
            }
            audit_and_require_sufficient_candidates(
                output_root=output_root,
                candidates=candidates,
                failures=all_failures,
                sources=selected_sources,
                required=config.data.samples_per_source,
                report=report_identity,
            )

            headers = build_v2_controlled_manifests(
                candidates,
                output_root=output_root,
                source_hashes=source_hashes,
                config_hash=locked.config_hash,
                seed=config.run.seed,
                samples_per_source=config.data.samples_per_source,
            )
            build_report = {
                **report_identity,
                "status": "complete",
                "required_counts_by_source": {
                    source: config.data.samples_per_source
                    for source in sorted(selected_sources)
                },
                "eligible_counts_by_source": {
                    source: reports[source]["unique_eligible_count"]
                    for source in sorted(selected_sources)
                },
                "failure_counts_by_source_and_kind": {
                    source: reports[source]["failure_counts"]
                    for source in selected_sources
                },
                "manifest_hashes": {
                    source: headers[source].manifest_hash
                    for source in selected_sources
                },
            }
            write_v2_build_report(output_root, build_report)
            return build_report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    build_manifests(
        args.config,
        candidate_pool=args.candidate_pool,
        sources=args.sources,
    )


if __name__ == "__main__":
    main()
