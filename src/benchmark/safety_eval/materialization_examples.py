"""Deterministic qualitative examples for materialization fidelity evidence."""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Iterable
from dataclasses import dataclass

from .intent_preservation import AnnotationMapping, DriftReason, FinalIntentLabel, FinalLabel
from .materialization_ablation import Branch, MaterializationPair, index_pairs


@dataclass(frozen=True)
class ExampleCase:
    case_id: str
    source: str
    sample_id: str
    branch: Branch
    reference_intent: str
    initial_discrete_prompt: str
    materialized_prompt: str
    final_label: FinalLabel
    drift_reasons: tuple[DriftReason, ...]
    short_explanation: str
    roundtrip_exact_match: bool
    mean_projection_cosine: float


@dataclass(frozen=True)
class ExampleSlot:
    source: str
    branch: Branch
    final_label: FinalLabel
    case: ExampleCase | None


_REASON_EXPLANATIONS = {
    DriftReason.action_changed: "The materialized prompt changes the core requested action.",
    DriftReason.target_changed: "The materialized prompt changes the target or affected entity.",
    DriftReason.constraint_dropped: "The materialized prompt drops a constraint that changes the task.",
    DriftReason.contradiction_added: "The materialized prompt adds content that contradicts the reference intent.",
    DriftReason.uninterpretable: "The materialized prompt is not sufficiently interpretable to preserve the intent.",
    DriftReason.other: "The adjudicated annotation records another material change to the reference intent.",
}


def _explanation(label: FinalIntentLabel) -> str:
    if label.final_label is FinalLabel.preserved:
        return "The core action, target, and task-defining constraints remain recognizable under the annotation criteria."
    return " ".join(_REASON_EXPLANATIONS[reason] for reason in label.drift_reasons)


def build_example_cases(
    pairs: Iterable[MaterializationPair],
    mappings: Iterable[AnnotationMapping],
    labels: Iterable[FinalIntentLabel],
    *,
    sources: tuple[str, ...],
) -> tuple[ExampleSlot, ...]:
    """Select the first sample ID in every fixed source/branch/label stratum."""
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("example sources must be non-empty and unique")
    pair_index = index_pairs(pairs)
    mapping_rows = tuple(mappings)
    label_rows = tuple(labels)
    mapping_by_id = {row.annotation_id: row for row in mapping_rows}
    label_by_id = {row.annotation_id: row for row in label_rows}
    if len(mapping_by_id) != len(mapping_rows) or len(label_by_id) != len(label_rows):
        raise ValueError("duplicate annotation identity in example inputs")
    if set(mapping_by_id) != set(label_by_id):
        raise ValueError("example labels do not match annotation mappings")
    candidates: dict[tuple[str, Branch, FinalLabel], list[tuple[MaterializationPair, FinalIntentLabel]]] = {}
    for annotation_id, mapping in mapping_by_id.items():
        key = mapping.source, mapping.sample_id, mapping.branch, mapping.optimization_checkpoint
        pair = pair_index.get(key)
        if pair is None or pair.status != "complete":
            raise ValueError(f"example mapping references an unavailable pair: {key}")
        if hashlib.sha256(pair.materialized_text.encode("utf-8")).hexdigest() != mapping.materialized_text_hash:
            raise ValueError(f"example prompt hash mismatch: {annotation_id}")
        label = label_by_id[annotation_id]
        candidates.setdefault((mapping.source, mapping.branch, label.final_label), []).append((pair, label))

    slots: list[ExampleSlot] = []
    case_number = 0
    for source in sources:
        for branch in (Branch.high_value, Branch.safety_sensitivity):
            for final_label in (FinalLabel.preserved, FinalLabel.not_preserved):
                case_number += 1
                rows = sorted(candidates.get((source, branch, final_label), ()), key=lambda row: row[0].sample_id)
                if not rows:
                    slots.append(ExampleSlot(source, branch, final_label, None))
                    continue
                pair, label = rows[0]
                slots.append(ExampleSlot(source, branch, final_label, ExampleCase(
                    case_id=f"case_{case_number:02d}",
                    source=source,
                    sample_id=pair.sample_id,
                    branch=branch,
                    reference_intent=pair.reference_intent,
                    initial_discrete_prompt=pair.initial_discrete_prompt,
                    materialized_prompt=pair.materialized_text,
                    final_label=final_label,
                    drift_reasons=label.drift_reasons,
                    short_explanation=_explanation(label),
                    roundtrip_exact_match=pair.roundtrip_exact_match,
                    mean_projection_cosine=sum(pair.projection_cosines) / len(pair.projection_cosines),
                )))
    return tuple(slots)


def select_compact_examples(slots: Iterable[ExampleSlot]) -> tuple[ExampleCase, ...]:
    """Select the shortest two-label set after maximizing source/branch coverage."""
    cases = tuple(slot.case for slot in slots if slot.case is not None)
    if not cases:
        return ()
    preserved = tuple(case for case in cases if case.final_label is FinalLabel.preserved)
    drifted = tuple(case for case in cases if case.final_label is FinalLabel.not_preserved)
    if not preserved or not drifted:
        return (min(cases, key=lambda case: (len(case.materialized_prompt), case.case_id)),)
    combinations = itertools.product(preserved, drifted)
    selected = min(
        combinations,
        key=lambda pair: (
            -len({case.source for case in pair}),
            -len({case.branch for case in pair}),
            sum(len(case.materialized_prompt) for case in pair),
            tuple(case.case_id for case in pair),
        ),
    )
    return tuple(sorted(selected, key=lambda case: case.case_id))


def _quote(value: str) -> str:
    return "\n".join(f"> {line}" for line in value.splitlines())


def render_examples_markdown(slots: Iterable[ExampleSlot], *, title: str) -> str:
    """Render an index and verbatim case sections without content-based selection."""
    rows = tuple(slots)
    lines = [f"# {title}", "", "| Case | Source | Branch | Intent label | Drift reason | Round trip |", "|---|---|---|---|---|---|"]
    for slot in rows:
        case = slot.case
        if case is None:
            lines.append(
                f"| No case available | {slot.source} | {slot.branch.value} | {slot.final_label.value} | - | - |"
            )
        else:
            reasons = ", ".join(reason.value for reason in case.drift_reasons) or "None"
            roundtrip = "Exact" if case.roundtrip_exact_match else "Changed"
            lines.append(f"| {case.case_id} | {case.source} | {case.branch.value} | {case.final_label.value} | {reasons} | {roundtrip} |")
    for slot in rows:
        case = slot.case
        if case is None:
            continue
        reasons = ", ".join(reason.value for reason in case.drift_reasons) or "None"
        roundtrip = "Exact" if case.roundtrip_exact_match else "Changed"
        lines.extend([
            "",
            f"## Case {case.case_id}",
            "",
            f"- Source: {case.source}",
            f"- Sample ID: {case.sample_id}",
            f"- Branch: {case.branch.value}",
            f"- Human intent label: {case.final_label.value}",
            f"- Drift reason: {reasons}",
            f"- Token round trip: {roundtrip}",
            f"- Mean projection cosine: {case.mean_projection_cosine:.6f}",
            "",
            "**Reference intent**",
            "",
            _quote(case.reference_intent),
            "",
            "**Initial discrete prompt**",
            "",
            _quote(case.initial_discrete_prompt),
            "",
            "**Materialized prompt**",
            "",
            _quote(case.materialized_prompt),
            "",
            "**Assessment**",
            "",
            case.short_explanation,
        ])
    return "\n".join(lines) + "\n"
