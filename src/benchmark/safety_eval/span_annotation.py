"""Offline model annotation of editable prompt spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import Field, model_validator

from .config import AnnotationConfig
from .io import canonical_json
from .schema import EditableSpan, EditableSpanRole, Sha256, StrictRecord


_SPAN_FIELDS = frozenset(
    {"start", "end", "quote", "role", "confidence", "rationale"}
)
_ROLE_VALUES = tuple(sorted(role.value for role in EditableSpanRole))


@runtime_checkable
class AnnotationTransport(Protocol):
    model: str
    revision: str

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
    ) -> str: ...


class SpanAnnotationError(ValueError):
    """Raised when an annotation cannot satisfy the locked contract."""


class LowConfidenceSpanAnnotationError(SpanAnnotationError):
    """Raised when a valid annotation is below the frozen confidence threshold."""


class FrozenSpanAnnotation(StrictRecord):
    spans: tuple[EditableSpan, ...] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    template_sha256: Sha256
    response_sha256: Sha256
    model: str
    revision: str

    @model_validator(mode="after")
    def validate_confidence(self) -> "FrozenSpanAnnotation":
        if self.confidence != min(span.confidence for span in self.spans):
            raise ValueError("confidence must equal the minimum span confidence")
        if not self.model.strip() or not self.revision.strip():
            raise ValueError("annotation model and revision must be non-empty")
        return self


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


def _schema_error(response: str, prompt: str) -> tuple[EditableSpan, ...]:
    try:
        payload = json.loads(response, object_pairs_hook=_unique_object)
    except _DuplicateJsonField as error:
        raise SpanAnnotationError(
            f"response contains duplicate JSON field: {error.field}"
        ) from error
    except json.JSONDecodeError as error:
        raise SpanAnnotationError("response must be valid JSON") from error
    if not isinstance(payload, dict):
        raise SpanAnnotationError("response must be a JSON object")

    unexpected = sorted(set(payload) - {"spans"})
    if unexpected:
        raise SpanAnnotationError(
            f"response has unexpected fields: {', '.join(unexpected)}"
        )
    if "spans" not in payload:
        raise SpanAnnotationError("response is missing required field: spans")
    raw_spans = payload["spans"]
    if not isinstance(raw_spans, list):
        raise SpanAnnotationError("spans must be an array")
    if not raw_spans:
        raise SpanAnnotationError("spans must contain at least one item")

    spans: list[EditableSpan] = []
    previous_end = 0
    for index, raw_span in enumerate(raw_spans):
        prefix = f"spans[{index}]"
        if not isinstance(raw_span, dict):
            raise SpanAnnotationError(f"{prefix} must be an object")
        unexpected = sorted(set(raw_span) - _SPAN_FIELDS)
        if unexpected:
            raise SpanAnnotationError(
                f"{prefix} has unexpected fields: {', '.join(unexpected)}"
            )
        missing = sorted(_SPAN_FIELDS - set(raw_span))
        if missing:
            raise SpanAnnotationError(
                f"{prefix} is missing required fields: {', '.join(missing)}"
            )

        start = raw_span["start"]
        end = raw_span["end"]
        quote = raw_span["quote"]
        role = raw_span["role"]
        confidence = raw_span["confidence"]
        rationale = raw_span["rationale"]
        if type(start) is not int or start < 0:
            raise SpanAnnotationError(f"{prefix}.start must be a non-negative integer")
        if type(end) is not int or end <= start:
            raise SpanAnnotationError(
                f"{prefix}.end must be an integer greater than start"
            )
        if not isinstance(quote, str) or not quote:
            raise SpanAnnotationError(f"{prefix}.quote must be a non-empty string")
        if role not in _ROLE_VALUES:
            raise SpanAnnotationError(
                f"{prefix}.role must be one of: {', '.join(_ROLE_VALUES)}"
            )
        if type(confidence) not in (int, float):
            raise SpanAnnotationError(f"{prefix}.confidence must be a number")
        if not 0.0 <= confidence <= 1.0:
            raise SpanAnnotationError(f"{prefix}.confidence must be between 0 and 1")
        if not isinstance(rationale, str):
            raise SpanAnnotationError(f"{prefix}.rationale must be a string")
        if end > len(prompt):
            raise SpanAnnotationError(f"{prefix}.end must be within the prompt")
        if prompt[start:end] != quote:
            raise SpanAnnotationError(
                f"{prefix}.quote must equal prompt[start:end]"
            )
        if index and start < previous_end:
            raise SpanAnnotationError("spans must be ordered and non-overlapping")

        spans.append(
            EditableSpan(
                start=start,
                end=end,
                quote=quote,
                role=role,
                confidence=float(confidence),
                rationale=rationale,
            )
        )
        previous_end = end
    return tuple(spans)


def _repair_message(error: SpanAnnotationError) -> str:
    return (
        "Your response failed validation:\n"
        f"{error}\n"
        "Return one corrected JSON object matching the required schema. JSON only."
    )


class SpanAnnotator:
    """Apply the locked model annotator with one deterministic schema repair."""

    def __init__(
        self,
        config: AnnotationConfig,
        transport: AnnotationTransport,
        *,
        confidence_threshold: float,
    ) -> None:
        if type(confidence_threshold) not in (int, float) or not (
            0.0 <= confidence_threshold <= 1.0
        ):
            raise ValueError("confidence_threshold must be a number between 0 and 1")
        if transport.model != config.model or transport.revision != config.revision:
            raise SpanAnnotationError(
                "transport identity does not match the locked annotation config"
            )

        self._config = config
        self._transport = transport
        self._confidence_threshold = float(confidence_threshold)
        template_bytes = Path(config.template_path).read_bytes()
        try:
            self._template = template_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SpanAnnotationError("annotation template must be UTF-8") from error
        self._template_sha256 = hashlib.sha256(template_bytes).hexdigest()

    def annotate(self, prompt: str) -> FrozenSpanAnnotation:
        if not isinstance(prompt, str) or not prompt:
            raise SpanAnnotationError("prompt must be a non-empty string")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._template},
            {"role": "user", "content": canonical_json({"prompt": prompt})},
        ]
        first_response = self._transport.complete(
            messages,
            temperature=self._config.temperature,
        )
        try:
            spans = _schema_error(first_response, prompt)
            accepted_response = first_response
        except SpanAnnotationError as first_error:
            repair_messages = [
                *messages,
                {"role": "assistant", "content": first_response},
                {"role": "user", "content": _repair_message(first_error)},
            ]
            repaired_response = self._transport.complete(
                repair_messages,
                temperature=self._config.temperature,
            )
            try:
                spans = _schema_error(repaired_response, prompt)
            except SpanAnnotationError as second_error:
                raise SpanAnnotationError(
                    f"annotation remained invalid after one repair: {second_error}"
                ) from second_error
            accepted_response = repaired_response

        confidence = min(span.confidence for span in spans)
        if confidence < self._confidence_threshold:
            raise LowConfidenceSpanAnnotationError(
                f"minimum span confidence {confidence:g} is below frozen threshold "
                f"{self._confidence_threshold:g}"
            )
        return FrozenSpanAnnotation(
            spans=spans,
            confidence=confidence,
            template_sha256=self._template_sha256,
            response_sha256=hashlib.sha256(
                accepted_response.encode("utf-8")
            ).hexdigest(),
            model=self._transport.model,
            revision=self._transport.revision,
        )
