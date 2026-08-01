"""Deterministic tokenizer vocabulary policies for continuous projection."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
import re
import string
from typing import Any

from .io import canonical_hash


WORDFREQ_VERSION = "3.1.1"
ENGLISH_LANGUAGE = "en"
MIN_ENGLISH_ZIPF = 3.5
COMMON_TOKEN_ID_CEILING = 50_000
PROJECTION_TOKEN_POLICIES = (
    "special_only",
    "ascii_printable",
    "english_common_positioned",
)


@dataclass(frozen=True)
class PositionProjectionMask:
    original_token_id: int
    position_class: str
    allowed_token_ids: tuple[int, ...]
    allowed_token_count: int
    allowed_token_ids_sha256: str

    def evidence(self) -> dict[str, object]:
        return {
            "original_token_id": self.original_token_id,
            "position_class": self.position_class,
            "allowed_token_count": self.allowed_token_count,
            "allowed_token_ids_sha256": self.allowed_token_ids_sha256,
        }


@dataclass(frozen=True)
class ProjectionVocabulary:
    policy: str
    vocabulary_size: int
    allowed_token_ids: tuple[int, ...]
    allowed_token_count: int
    excluded_token_count: int
    allowed_token_ids_sha256: str
    common_english_token_ids: tuple[int, ...] = ()
    common_english_token_count: int | None = None
    common_english_token_ids_sha256: str | None = None
    z_position_masks: tuple[PositionProjectionMask, ...] = ()
    u_position_masks: tuple[PositionProjectionMask, ...] = ()
    position_mask_manifest_sha256: str | None = None
    wordfreq_version: str | None = None
    english_language: str | None = None
    minimum_english_zipf: float | None = None
    common_token_id_ceiling: int | None = None

    def evidence(self) -> dict[str, object]:
        evidence: dict[str, object] = {
            "policy": self.policy,
            "vocabulary_size": self.vocabulary_size,
            "allowed_token_count": self.allowed_token_count,
            "excluded_token_count": self.excluded_token_count,
            "allowed_token_ids_sha256": self.allowed_token_ids_sha256,
        }
        if self.policy == "english_common_positioned":
            evidence.update({
                "wordfreq_version": self.wordfreq_version,
                "english_language": self.english_language,
                "minimum_english_zipf": self.minimum_english_zipf,
                "common_token_id_ceiling": self.common_token_id_ceiling,
                "common_english_token_count": self.common_english_token_count,
                "common_english_token_ids_sha256": self.common_english_token_ids_sha256,
                "z_position_masks": [mask.evidence() for mask in self.z_position_masks],
                "u_position_masks": [mask.evidence() for mask in self.u_position_masks],
                "position_mask_manifest_sha256": self.position_mask_manifest_sha256,
            })
        return evidence


def classify_projection_piece(piece: str) -> str:
    if re.fullmatch(r" [a-z]+", piece):
        return "word_start"
    if re.fullmatch(r"[A-Z][a-z]*", piece):
        return "sentence_initial"
    if re.fullmatch(r"[a-z]+", piece):
        return "continuation"
    if re.fullmatch(r"'[a-z]+", piece):
        return "contraction"
    if piece and all(character in string.punctuation for character in piece):
        return "punctuation"
    return "other"


def _ascii_printable_piece(text: str) -> bool:
    return bool(text) and "\ufffd" not in text and all(
        32 <= ord(character) <= 126 or character in "\t\n\r"
        for character in text
    )


def _decode_piece(tokenizer: Any, token_id: int) -> str:
    return str(tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ))


def _resolve_word_frequency(
    zipf_frequency: Callable[[str, str], float] | None,
    wordfreq_version: str | None,
) -> tuple[Callable[[str, str], float], str]:
    try:
        resolved_version = (
            package_version("wordfreq")
            if wordfreq_version is None
            else wordfreq_version
        )
    except PackageNotFoundError as error:
        raise RuntimeError("wordfreq package is unavailable") from error
    if resolved_version != WORDFREQ_VERSION:
        raise RuntimeError(
            f"wordfreq version must be {WORDFREQ_VERSION}, got {resolved_version}"
        )
    if zipf_frequency is None:
        try:
            from wordfreq import zipf_frequency as installed_zipf_frequency
        except ImportError as error:
            raise RuntimeError("wordfreq package is unavailable") from error
        zipf_frequency = installed_zipf_frequency
    return zipf_frequency, resolved_version


def _editable_token_ids(
    values: Iterable[int] | None,
    *,
    block: str,
    vocabulary_size: int,
) -> tuple[int, ...]:
    if values is None:
        raise ValueError(
            f"{block}_token_ids are required for english_common_positioned"
        )
    ordered = tuple(values)
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        or token_id >= vocabulary_size
        for token_id in ordered
    ):
        raise ValueError(f"{block}_token_ids contain an invalid token ID")
    return ordered


def _position_mask(
    tokenizer: Any,
    original_token_id: int,
    common_token_ids: tuple[int, ...],
) -> PositionProjectionMask:
    position_class = classify_projection_piece(
        _decode_piece(tokenizer, original_token_id)
    )
    if position_class == "word_start":
        allowed_token_ids = common_token_ids
        if original_token_id not in allowed_token_ids:
            allowed_token_ids += (original_token_id,)
    else:
        allowed_token_ids = (original_token_id,)
    return PositionProjectionMask(
        original_token_id=original_token_id,
        position_class=position_class,
        allowed_token_ids=allowed_token_ids,
        allowed_token_count=len(allowed_token_ids),
        allowed_token_ids_sha256=canonical_hash(list(allowed_token_ids)),
    )


def _full_position_manifest(
    z_position_masks: tuple[PositionProjectionMask, ...],
    u_position_masks: tuple[PositionProjectionMask, ...],
) -> dict[str, list[dict[str, object]]]:
    def full_evidence(mask: PositionProjectionMask) -> dict[str, object]:
        return {
            "original_token_id": mask.original_token_id,
            "position_class": mask.position_class,
            "allowed_token_ids": list(mask.allowed_token_ids),
        }

    return {
        "z_position_masks": [full_evidence(mask) for mask in z_position_masks],
        "u_position_masks": [full_evidence(mask) for mask in u_position_masks],
    }


def _build_english_common_vocabulary(
    tokenizer: Any,
    vocabulary_size: int,
    special_ids: set[int],
    *,
    z_token_ids: Iterable[int] | None,
    u_token_ids: Iterable[int] | None,
    zipf_frequency: Callable[[str, str], float] | None,
    wordfreq_version: str | None,
) -> ProjectionVocabulary:
    z_ids = _editable_token_ids(
        z_token_ids,
        block="z",
        vocabulary_size=vocabulary_size,
    )
    u_ids = _editable_token_ids(
        u_token_ids,
        block="u",
        vocabulary_size=vocabulary_size,
    )
    frequency, resolved_version = _resolve_word_frequency(
        zipf_frequency,
        wordfreq_version,
    )
    common_ids: list[int] = []
    for token_id in range(min(vocabulary_size, COMMON_TOKEN_ID_CEILING)):
        if token_id in special_ids:
            continue
        piece = _decode_piece(tokenizer, token_id)
        if not re.fullmatch(r" [a-z]+", piece):
            continue
        if frequency(piece[1:], ENGLISH_LANGUAGE) >= MIN_ENGLISH_ZIPF:
            common_ids.append(token_id)
    if not common_ids:
        raise ValueError("projection token policy leaves no common-English tokens")

    ordered = tuple(common_ids)
    z_masks = tuple(_position_mask(tokenizer, token_id, ordered) for token_id in z_ids)
    u_masks = tuple(_position_mask(tokenizer, token_id, ordered) for token_id in u_ids)
    manifest = _full_position_manifest(z_masks, u_masks)
    return ProjectionVocabulary(
        policy="english_common_positioned",
        vocabulary_size=vocabulary_size,
        allowed_token_ids=ordered,
        allowed_token_count=len(ordered),
        excluded_token_count=vocabulary_size - len(ordered),
        allowed_token_ids_sha256=canonical_hash(list(ordered)),
        common_english_token_ids=ordered,
        common_english_token_count=len(ordered),
        common_english_token_ids_sha256=canonical_hash(list(ordered)),
        z_position_masks=z_masks,
        u_position_masks=u_masks,
        position_mask_manifest_sha256=canonical_hash(manifest),
        wordfreq_version=resolved_version,
        english_language=ENGLISH_LANGUAGE,
        minimum_english_zipf=MIN_ENGLISH_ZIPF,
        common_token_id_ceiling=COMMON_TOKEN_ID_CEILING,
    )


def build_projection_vocabulary(
    tokenizer: Any,
    vocabulary_size: int,
    policy: str,
    *,
    z_token_ids: Iterable[int] | None = None,
    u_token_ids: Iterable[int] | None = None,
    zipf_frequency: Callable[[str, str], float] | None = None,
    wordfreq_version: str | None = None,
) -> ProjectionVocabulary:
    if policy not in PROJECTION_TOKEN_POLICIES:
        raise ValueError(f"unsupported projection token policy: {policy}")
    if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int) or vocabulary_size < 1:
        raise ValueError("projection vocabulary size must be a positive integer")

    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    if policy == "english_common_positioned":
        return _build_english_common_vocabulary(
            tokenizer,
            vocabulary_size,
            special_ids,
            z_token_ids=z_token_ids,
            u_token_ids=u_token_ids,
            zipf_frequency=zipf_frequency,
            wordfreq_version=wordfreq_version,
        )

    allowed: list[int] = []
    for token_id in range(vocabulary_size):
        if token_id in special_ids:
            continue
        if policy == "ascii_printable":
            piece = _decode_piece(tokenizer, token_id)
            if not _ascii_printable_piece(piece):
                continue
        allowed.append(token_id)

    if not allowed:
        raise ValueError("projection token policy leaves no allowed tokens")
    ordered = tuple(allowed)
    return ProjectionVocabulary(
        policy=policy,
        vocabulary_size=vocabulary_size,
        allowed_token_ids=ordered,
        allowed_token_count=len(ordered),
        excluded_token_count=vocabulary_size - len(ordered),
        allowed_token_ids_sha256=canonical_hash(list(ordered)),
    )


def validate_initial_editable_ids(
    vocabulary: ProjectionVocabulary,
    *,
    z_token_ids: Iterable[int],
    u_token_ids: Iterable[int],
) -> None:
    z_ids = tuple(int(value) for value in z_token_ids)
    u_ids = tuple(int(value) for value in u_token_ids)
    if vocabulary.policy == "english_common_positioned":
        def matches_position_masks(
            token_ids: tuple[int, ...],
            masks: tuple[PositionProjectionMask, ...],
        ) -> bool:
            return len(token_ids) == len(masks) and all(
                token_id == mask.original_token_id
                and token_id in mask.allowed_token_ids
                for token_id, mask in zip(token_ids, masks, strict=True)
            )

        if not matches_position_masks(
            z_ids,
            vocabulary.z_position_masks,
        ) or not matches_position_masks(
            u_ids,
            vocabulary.u_position_masks,
        ):
            raise ValueError("projection policy excludes an initial editable token")
        return

    allowed = set(vocabulary.allowed_token_ids)
    initial_ids = z_ids + u_ids
    if any(token_id not in allowed for token_id in initial_ids):
        raise ValueError("projection policy excludes an initial editable token")
