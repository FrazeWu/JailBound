"""Deterministic offline annotation of editable prompt spans."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .schema import EditableSpan, EditableSpanRole


class AnnotationTransport(Protocol):
    """Injected completion interface used by the offline annotator."""

    model: str
    revision: str

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float,
    ) -> str: ...


class SpanAnnotationError(ValueError):
    """The model did not produce an acceptable editable-span annotation."""


@dataclass(frozen=True)
class FrozenSpanAnnotation:
    spans: tuple[EditableSpan, ...]
    confidence: float
    template_sha256: str
    response_sha256: str
    model: str
    revision: str
    offset_corrections: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class SpanAnnotationSettings:
    """Immutable annotation policy; sampling and repair count are fixed."""

    template: str
    confidence_threshold: float
    temperature: float = field(default=0.0, init=False)
    repair_attempts: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not self.template:
            raise ValueError("span annotation template cannot be empty")
        try:
            threshold = float(self.confidence_threshold)
        except (OverflowError, TypeError, ValueError):
            threshold = math.nan
        if (
            isinstance(self.confidence_threshold, bool)
            or not isinstance(self.confidence_threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("span annotation confidence threshold must be a finite probability")


class _StrictJSONError(ValueError):
    pass


def _reject_constant(value: str) -> None:
    raise _StrictJSONError(f"non-standard JSON constant {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJSONError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_response(raw_response: str) -> tuple[EditableSpan, ...]:
    if not isinstance(raw_response, str):
        raise _StrictJSONError("response must be a string")
    try:
        payload = json.loads(
            raw_response,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise _StrictJSONError(
            f"response is not valid JSON: {error.msg} at line {error.lineno} column {error.colno}"
        ) from error
    except ValueError as error:
        raise _StrictJSONError(f"response JSON contains an invalid numeric value: {error}") from error
    if not isinstance(payload, dict):
        raise _StrictJSONError("response must be a JSON object")
    if set(payload) != {"spans"}:
        raise _StrictJSONError("response object must contain exactly the key 'spans'")
    raw_spans = payload["spans"]
    if not isinstance(raw_spans, list) or not raw_spans:
        raise _StrictJSONError("spans must be a non-empty JSON array")

    spans: list[EditableSpan] = []
    expected_fields = {"start", "end", "quote", "role", "confidence", "rationale"}
    for index, item in enumerate(raw_spans):
        prefix = f"span[{index}]"
        if not isinstance(item, dict):
            raise _StrictJSONError(f"{prefix} must be a JSON object")
        if set(item) != expected_fields:
            raise _StrictJSONError(
                f"{prefix} must contain exactly start, end, quote, role, confidence, rationale"
            )
        start, end = item["start"], item["end"]
        if isinstance(start, bool) or not isinstance(start, int):
            raise _StrictJSONError(f"{prefix}.start must be an integer")
        if isinstance(end, bool) or not isinstance(end, int):
            raise _StrictJSONError(f"{prefix}.end must be an integer")
        if start < 0 or end <= start:
            raise _StrictJSONError(f"{prefix} requires non-empty ordered bounds")
        quote = item["quote"]
        if not isinstance(quote, str) or not quote:
            raise _StrictJSONError(f"{prefix}.quote must be a non-empty string")
        role = item["role"]
        if not isinstance(role, str):
            raise _StrictJSONError(f"{prefix}.role must be a string")
        try:
            span_role = EditableSpanRole(role)
        except ValueError as error:
            raise _StrictJSONError(f"{prefix}.role is not an editable span role") from error
        confidence = item["confidence"]
        try:
            numeric_confidence = float(confidence)
        except (OverflowError, TypeError, ValueError):
            numeric_confidence = math.nan
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(numeric_confidence)
            or not 0.0 <= numeric_confidence <= 1.0
        ):
            raise _StrictJSONError(f"{prefix}.confidence must be a finite probability")
        rationale = item["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise _StrictJSONError(f"{prefix}.rationale must be a non-empty string")
        spans.append(
            EditableSpan(
                start=start,
                end=end,
                quote=quote,
                role=span_role,
                confidence=numeric_confidence,
                rationale=rationale,
            )
        )
    return tuple(spans)


def _validate_offsets(full_text: str, spans: tuple[EditableSpan, ...]) -> None:
    previous: EditableSpan | None = None
    for index, span in enumerate(spans):
        if previous is not None and span.start < previous.end:
            raise _StrictJSONError(f"span[{index}] overlaps or precedes span[{index - 1}]")
        if span.end > len(full_text) or full_text[span.start : span.end] != span.quote:
            raise _StrictJSONError(
                f"span[{index}].quote does not match full_text at the declared offsets"
            )
        previous = span


def _parse_annotation(raw_response: str, full_text: str) -> tuple[EditableSpan, ...]:
    spans = _decode_response(raw_response)
    _validate_offsets(full_text, spans)
    return spans


def _correct_unique_quote_offsets(
    raw_response: str, full_text: str
) -> tuple[tuple[EditableSpan, ...], tuple[dict[str, object], ...]]:
    spans = _decode_response(raw_response)
    corrected: list[EditableSpan] = []
    corrections: list[dict[str, object]] = []
    for index, span in enumerate(spans):
        if span.end <= len(full_text) and full_text[span.start : span.end] == span.quote:
            corrected.append(span)
            continue
        start = full_text.find(span.quote)
        if start < 0 or full_text.find(span.quote, start + 1) >= 0:
            raise _StrictJSONError(
                f"span[{index}].quote does not have one unique exact offset candidate"
            )
        end = start + len(span.quote)
        corrected.append(span.model_copy(update={"start": start, "end": end}))
        corrections.append({
            "span": index,
            "quote": span.quote,
            "declared_bounds": [span.start, span.end],
            "corrected_bounds": [start, end],
        })
    result = tuple(corrected)
    _validate_offsets(full_text, result)
    return result, tuple(corrections)


def _input_message(
    full_text: str,
    seed_intent: str,
    source_hints: Sequence[str],
) -> str:
    compact_ascii = len(full_text) > 512 and full_text.isascii()
    payload = {
        "full_text": full_text,
        "seed_intent": seed_intent,
        "source_hints": list(source_hints),
        "unicode_codepoint_index": (
            [] if compact_ascii
            else [[index, character] for index, character in enumerate(full_text)]
        ),
        "offset_contract": (
            "ASCII; Python code-point offsets equal byte-independent character offsets"
            if compact_ascii else "unicode_codepoint_index is authoritative"
        ),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _exact_offset_candidates(full_text: str, raw_response: str) -> list[dict[str, object]]:
    try:
        spans = _decode_response(raw_response)
    except _StrictJSONError:
        return []
    corrections: list[dict[str, object]] = []
    for index, span in enumerate(spans):
        bounds: list[list[int]] = []
        start = 0
        while True:
            match = full_text.find(span.quote, start)
            if match < 0:
                break
            bounds.append([match, match + len(span.quote)])
            start = match + 1
        if bounds:
            corrections.append({
                "span": index,
                "quote": span.quote,
                "candidate_bounds": bounds,
            })
    return corrections


def _repair_message(error: str, *, full_text: str, raw_response: str) -> str:
    message = (
        "Your response was invalid. Return one corrected JSON object and no other text.\n"
        f"Validation errors:\n- {error}"
    )
    candidates = _exact_offset_candidates(full_text, raw_response)
    if candidates:
        message += "\nExact offset candidates:\n" + json.dumps(
            candidates, ensure_ascii=False, separators=(",", ":")
        )
    return message


class SpanAnnotator:
    """Annotate exact editable character spans with one deterministic repair."""

    def __init__(
        self,
        transport: AnnotationTransport,
        settings: SpanAnnotationSettings,
    ) -> None:
        self.transport = transport
        self.settings = settings

    def annotate(
        self,
        full_text: str,
        seed_intent: str,
        source_hints: Sequence[str],
    ) -> FrozenSpanAnnotation:
        messages: tuple[Mapping[str, str], ...] = (
            {"role": "system", "content": self.settings.template},
            {
                "role": "user",
                "content": _input_message(full_text, seed_intent, source_hints),
            },
        )
        raw_response = self.transport.complete(
            messages,
            temperature=self.settings.temperature,
        )
        offset_corrections: tuple[dict[str, object], ...] = ()
        try:
            spans = _parse_annotation(raw_response, full_text)
        except _StrictJSONError as first_error:
            repair_messages = messages + (
                {"role": "assistant", "content": raw_response},
                {
                    "role": "user",
                    "content": _repair_message(
                        str(first_error), full_text=full_text, raw_response=raw_response
                    ),
                },
            )
            raw_response = self.transport.complete(
                repair_messages,
                temperature=self.settings.temperature,
            )
            try:
                spans = _parse_annotation(raw_response, full_text)
            except _StrictJSONError as second_error:
                try:
                    spans, offset_corrections = _correct_unique_quote_offsets(
                        raw_response, full_text
                    )
                except _StrictJSONError as correction_error:
                    raise SpanAnnotationError(
                        "span annotation response invalid after one repair attempt: "
                        f"{correction_error}"
                    ) from second_error

        confidence = min(span.confidence for span in spans)
        if confidence < self.settings.confidence_threshold:
            raise SpanAnnotationError(
                "span annotation confidence "
                f"{confidence} is below threshold {self.settings.confidence_threshold}"
            )
        return FrozenSpanAnnotation(
            spans=spans,
            confidence=confidence,
            template_sha256=hashlib.sha256(self.settings.template.encode("utf-8")).hexdigest(),
            response_sha256=hashlib.sha256(raw_response.encode("utf-8")).hexdigest(),
            model=self.transport.model,
            revision=self.transport.revision,
            offset_corrections=offset_corrections,
        )
