"""Prepare human span reviews and freeze their calibrated confidence threshold."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from benchmark.safety_eval.config import AnnotationConfig, load_v2_config
from benchmark.safety_eval.datasets import RawExample, load_source
from benchmark.safety_eval.io import atomic_write_jsonl, canonical_json
from benchmark.safety_eval.span_annotation import (
    AnnotationTransport,
    FrozenSpanAnnotation,
    SpanAnnotator,
    SpanAnnotationError,
)

ARTIFACT_VERSION = "span_annotation_confidence.v1"
FREEZE_COMMAND = "calibrate_span_annotation_confidence.py freeze"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ANNOTATION_RESPONSE_FORMAT: dict[str, object] = {
    "type": "json_schema",
    "json_schema": {
        "name": "editable_span_annotation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "spans": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "integer", "minimum": 0},
                            "end": {"type": "integer", "minimum": 1},
                            "quote": {"type": "string", "minLength": 1},
                            "role": {
                                "type": "string",
                                "enum": [
                                    "seed_intent",
                                    "harmful_payload",
                                    "attack_instruction",
                                ],
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "start",
                            "end",
                            "quote",
                            "role",
                            "confidence",
                            "rationale",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["spans"],
            "additionalProperties": False,
        },
    },
}


class CalibrationError(ValueError):
    """Raised when reviewed annotations cannot yield a frozen threshold."""


class _DuplicateJsonField(ValueError):
    def __init__(self, field: str) -> None:
        self.field = field


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonField(key)
        result[key] = value
    return result


@dataclass(frozen=True)
class ReviewedAnnotation:
    confidence: float
    accepted: bool | None


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    selected_count: int
    accepted_count: int
    reviewed_count: int
    precision: float
    recall: float
    target_precision: float
    minimum_selected: int


class Annotator(Protocol):
    def annotate(
        self,
        prompt: str,
        *,
        seed_intent: str,
        source_hints: Mapping[str, object],
    ) -> FrozenSpanAnnotation: ...


def _validated_review(value: object, index: int) -> ReviewedAnnotation:
    if isinstance(value, ReviewedAnnotation):
        review = value
    elif isinstance(value, Mapping):
        review = ReviewedAnnotation(
            confidence=value.get("confidence"),  # type: ignore[arg-type]
            accepted=value.get("accepted"),  # type: ignore[arg-type]
        )
    else:
        raise CalibrationError(f"review {index} must be a reviewed annotation object")
    if type(review.confidence) not in (int, float) or not math.isfinite(
        review.confidence
    ) or not 0.0 <= review.confidence <= 1.0:
        raise CalibrationError(f"review {index} confidence must be between 0 and 1")
    if review.accepted is None:
        raise CalibrationError(f"review {index} is incomplete")
    if type(review.accepted) is not bool:
        raise CalibrationError(f"review {index} accepted must be an exact boolean")
    return ReviewedAnnotation(float(review.confidence), review.accepted)


def calibrate_confidence(
    reviews: Sequence[object],
    *,
    target_precision: float,
    minimum_selected: int,
) -> CalibrationResult:
    """Select the maximum-recall qualifying observed confidence threshold."""
    if type(target_precision) not in (int, float) or not math.isfinite(
        target_precision
    ) or not 0.0 < target_precision <= 1.0:
        raise ValueError("target_precision must be a number in (0, 1]")
    if type(minimum_selected) is not int or minimum_selected < 1:
        raise ValueError("minimum_selected must be a positive integer")

    validated = tuple(
        _validated_review(review, index) for index, review in enumerate(reviews)
    )
    if not validated:
        raise CalibrationError("reviews cannot be empty")
    accepted_count = sum(review.accepted is True for review in validated)
    if accepted_count == 0:
        raise CalibrationError("at least one accepted positive is required for recall")

    candidates: list[tuple[float, int, float, float]] = []
    for threshold in sorted({review.confidence for review in validated}):
        selected = tuple(
            review for review in validated if review.confidence >= threshold
        )
        selected_accepted = sum(review.accepted is True for review in selected)
        precision = selected_accepted / len(selected)
        recall = selected_accepted / accepted_count
        if len(selected) >= minimum_selected and precision >= target_precision:
            candidates.append((threshold, len(selected), precision, recall))
    if not candidates:
        raise CalibrationError("no observed confidence threshold qualifies")

    threshold, selected_count, precision, recall = max(
        candidates, key=lambda item: (item[3], -item[0])
    )
    return CalibrationResult(
        threshold=threshold,
        selected_count=selected_count,
        accepted_count=accepted_count,
        reviewed_count=len(validated),
        precision=precision,
        recall=recall,
        target_precision=float(target_precision),
        minimum_selected=minimum_selected,
    )


def _sample_key(seed: int, row: RawExample) -> str:
    return hashlib.sha256(f"{seed}|{row.source_row_id}".encode("utf-8")).hexdigest()


def _annotation_payload_sha256(row: Mapping[str, object]) -> str:
    payload = {
        key: value
        for key, value in row.items()
        if key != "annotation_payload_sha256"
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def prepare_review_rows(
    sources: Mapping[str, Sequence[RawExample]],
    *,
    per_source: int,
    seed: int,
    annotator: Annotator,
    failures: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Deterministically sample and annotate a fixed number from every source."""
    if type(per_source) is not int or per_source < 1:
        raise ValueError("per_source must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rows: list[dict[str, object]] = []
    for source in sorted(sources):
        candidates = tuple(sources[source])
        if len(candidates) < per_source:
            raise CalibrationError(
                f"source {source!r} requires {per_source} rows but has {len(candidates)}"
            )
        chosen = sorted(candidates, key=lambda row: _sample_key(seed, row))[
            :per_source
        ]
        for raw in chosen:
            if raw.source != source:
                raise CalibrationError(
                    f"source mapping {source!r} contains row from {raw.source!r}"
                )
            source_hints: dict[str, object] = {
                "source": source,
                "source_row": raw.source_row,
                "source_row_id": raw.source_row_id,
                "attack_label": raw.source_attack_label,
                "domain_label": raw.source_domain_label,
                "language": raw.language,
                "risk_label": raw.source_risk_label,
                "preprocessing": list(raw.preprocessing),
            }
            try:
                annotation = annotator.annotate(
                    raw.attack_text,
                    seed_intent=raw.intent,
                    source_hints=source_hints,
                )
            except SpanAnnotationError as error:
                if failures is None:
                    raise
                failures.append(
                    {
                        "source": source,
                        "source_row": raw.source_row,
                        "source_row_id": raw.source_row_id,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                continue
            row: dict[str, object] = {
                "source": source,
                "source_row": raw.source_row,
                "source_row_id": raw.source_row_id,
                "prompt": raw.attack_text,
                "seed_intent": raw.intent,
                "source_hints": source_hints,
                "preprocessing": list(raw.preprocessing),
                "spans": [span.model_dump(mode="json") for span in annotation.spans],
                "confidence": annotation.confidence,
                "annotation_model": annotation.model,
                "annotation_revision": annotation.revision,
                "annotation_template_sha256": annotation.template_sha256,
                "annotation_response_sha256": annotation.response_sha256,
                "accepted": True,
            }
            row["annotation_payload_sha256"] = _annotation_payload_sha256(row)
            rows.append(row)
    return rows


def write_prepared_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    atomic_write_jsonl(path, rows)


def _read_reviewed_rows(path: Path) -> tuple[bytes, list[dict[str, object]]]:
    exact_bytes = path.read_bytes()
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(exact_bytes.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_unique_object)
        except _DuplicateJsonField as error:
            raise CalibrationError(
                f"reviewed JSONL line {line_number} contains duplicate JSON field: "
                f"{error.field}"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CalibrationError(
                f"reviewed JSONL line {line_number} must be valid UTF-8 JSON"
            ) from error
        if not isinstance(row, dict):
            raise CalibrationError(
                f"reviewed JSONL line {line_number} must be an object"
            )
        rows.append(row)
    if not rows:
        raise CalibrationError("reviewed JSONL cannot be empty")
    return exact_bytes, rows


def _required_string(row: Mapping[str, object], field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise CalibrationError(f"review {index} requires non-empty {field}")
    return value


def freeze_reviewed_annotations(
    reviewed_path: Path,
    *,
    target_precision: float,
    minimum_selected: int,
) -> dict[str, object]:
    exact_bytes, rows = _read_reviewed_rows(reviewed_path)
    reviews: list[ReviewedAnnotation] = []
    provenance: set[tuple[str, str, str, str]] = set()
    sources: set[str] = set()
    annotation_identity: tuple[str, str, str] | None = None
    for index, row in enumerate(rows):
        if type(row.get("accepted")) is not bool:
            raise CalibrationError(f"review {index} accepted must be an exact boolean")
        reviews.append(
            ReviewedAnnotation(
                confidence=row.get("confidence"),  # type: ignore[arg-type]
                accepted=row["accepted"],  # type: ignore[arg-type]
            )
        )
        source = _required_string(row, "source", index)
        model = _required_string(row, "annotation_model", index)
        revision = _required_string(row, "annotation_revision", index)
        template_sha256 = _required_string(
            row, "annotation_template_sha256", index
        )
        response_sha256 = _required_string(
            row, "annotation_response_sha256", index
        )
        payload_sha256 = _required_string(row, "annotation_payload_sha256", index)
        if (
            _SHA256.fullmatch(template_sha256) is None
            or _SHA256.fullmatch(response_sha256) is None
            or _SHA256.fullmatch(payload_sha256) is None
        ):
            raise CalibrationError(f"review {index} annotation hashes must be SHA-256")
        if payload_sha256 != _annotation_payload_sha256(row):
            raise CalibrationError(
                f"review {index} changed the immutable annotation payload"
            )
        current_identity = (model, revision, template_sha256)
        if annotation_identity is None:
            annotation_identity = current_identity
        elif current_identity != annotation_identity:
            raise CalibrationError(
                "all reviews must use the same model, revision, and template"
            )
        sources.add(source)
        provenance.add((model, revision, template_sha256, response_sha256))

    result = calibrate_confidence(
        reviews,
        target_precision=target_precision,
        minimum_selected=minimum_selected,
    )
    return {
        "accepted_count": result.accepted_count,
        "annotation_provenance": [
            {
                "model": model,
                "revision": revision,
                "response_sha256": response,
                "template_sha256": template,
            }
            for model, revision, template, response in sorted(provenance)
        ],
        "artifact_version": ARTIFACT_VERSION,
        "command": FREEZE_COMMAND,
        "minimum_selected": result.minimum_selected,
        "precision": result.precision,
        "recall": result.recall,
        "reviewed_count": result.reviewed_count,
        "reviewed_sha256": hashlib.sha256(exact_bytes).hexdigest(),
        "selected_count": result.selected_count,
        "sources": sorted(sources),
        "target_precision": result.target_precision,
        "threshold": result.threshold,
    }


def write_frozen_artifact(path: Path, artifact: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(artifact) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite frozen artifact: {path}"
            ) from error
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class OpenAIAnnotationTransport(AnnotationTransport):
    """Call a revision-pinned OpenAI-compatible deployment."""

    def __init__(
        self,
        *,
        model: str,
        revision: str,
        endpoint: str,
        api_key: str,
        seed: int,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.revision = revision
        if type(seed) is not int:
            raise ValueError("annotation seed must be an integer")
        self._seed = seed
        self._client = OpenAI(api_key=api_key, base_url=endpoint)

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
    ) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.revision,
                messages=[dict(message) for message in messages],
                temperature=temperature,
                max_tokens=512,
                seed=self._seed,
                response_format=_ANNOTATION_RESPONSE_FORMAT,
            )
        except Exception as error:
            raise SpanAnnotationError(
                f"annotation transport request failed: {error}"
            ) from error
        if getattr(response, "model", None) != self.revision:
            raise CalibrationError(
                "annotation transport returned model identity different from the "
                "locked revision"
            )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise CalibrationError("annotation transport returned no text")
        return content


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--per-source", type=int, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--failures-output", type=Path, required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--reviewed", type=Path, required=True)
    freeze.add_argument("--target-precision", type=float, required=True)
    freeze.add_argument("--minimum-selected", type=int, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _prepare(args: argparse.Namespace) -> None:
    config = load_v2_config(_rooted(args.config))
    annotation_payload = config.annotation.model_dump()
    annotation_payload["template_path"] = _rooted(config.annotation.template_path)
    annotation_config = AnnotationConfig.model_validate(annotation_payload)
    transport = OpenAIAnnotationTransport(
        model=annotation_config.model,
        revision=annotation_config.revision,
        endpoint=annotation_config.endpoint,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        seed=config.run.seed,
    )
    # Calibration must retain all valid scores before a threshold is frozen.
    annotator = SpanAnnotator(annotation_config, transport, confidence_threshold=0.0)
    sources: dict[str, Sequence[RawExample]] = {}
    for source in config.data.sources:
        targets = (
            _rooted(config.data.harmbench_targets_path)
            if source == "harmbench"
            else None
        )
        sources[source] = load_source(
            source, _rooted(config.data.paths[source]), targets
        )
    failures: list[dict[str, object]] = []
    rows = prepare_review_rows(
        sources,
        per_source=args.per_source,
        seed=config.run.seed,
        annotator=annotator,
        failures=failures,
    )
    if not rows:
        raise CalibrationError("all selected annotations failed validation")
    write_prepared_rows(args.output, rows)
    write_prepared_rows(args.failures_output, failures)


def _freeze(args: argparse.Namespace) -> None:
    artifact = freeze_reviewed_annotations(
        args.reviewed,
        target_precision=args.target_precision,
        minimum_selected=args.minimum_selected,
    )
    write_frozen_artifact(args.output, artifact)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "prepare":
        _prepare(args)
    else:
        _freeze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
