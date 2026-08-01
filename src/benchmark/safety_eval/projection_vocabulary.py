"""Deterministic tokenizer vocabulary policies for continuous projection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .io import canonical_hash


PROJECTION_TOKEN_POLICIES = ("special_only", "ascii_printable")


@dataclass(frozen=True)
class ProjectionVocabulary:
    policy: str
    vocabulary_size: int
    allowed_token_ids: tuple[int, ...]
    allowed_token_count: int
    excluded_token_count: int
    allowed_token_ids_sha256: str

    def evidence(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "vocabulary_size": self.vocabulary_size,
            "allowed_token_count": self.allowed_token_count,
            "excluded_token_count": self.excluded_token_count,
            "allowed_token_ids_sha256": self.allowed_token_ids_sha256,
        }


def _ascii_printable_piece(text: str) -> bool:
    return bool(text) and "\ufffd" not in text and all(
        32 <= ord(character) <= 126 or character in "\t\n\r"
        for character in text
    )


def build_projection_vocabulary(
    tokenizer: Any,
    vocabulary_size: int,
    policy: str,
) -> ProjectionVocabulary:
    if policy not in PROJECTION_TOKEN_POLICIES:
        raise ValueError(f"unsupported projection token policy: {policy}")
    if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int) or vocabulary_size < 1:
        raise ValueError("projection vocabulary size must be a positive integer")

    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    allowed: list[int] = []
    for token_id in range(vocabulary_size):
        if token_id in special_ids:
            continue
        if policy == "ascii_printable":
            piece = str(tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ))
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
    allowed = set(vocabulary.allowed_token_ids)
    initial_ids = tuple(int(value) for value in z_token_ids) + tuple(
        int(value) for value in u_token_ids
    )
    if any(token_id not in allowed for token_id in initial_ids):
        raise ValueError("projection policy excludes an initial editable token")
