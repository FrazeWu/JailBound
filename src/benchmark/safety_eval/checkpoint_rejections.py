"""Strict loading for manual checkpoint rejection ledgers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from .checkpoint_early_stop import BRANCH_ORDER


_EXPECTED_FIELDS = ("branch", "step", "state_sha256", "reason")
_VALID_BRANCHES = frozenset(BRANCH_ORDER)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ManualCheckpointRejection:
    branch: str
    step: int
    state_sha256: str
    reason: str

    @property
    def branch_step(self) -> tuple[str, int]:
        return self.branch, self.step

    def evidence(self) -> dict[str, object]:
        return {
            "branch": self.branch,
            "step": self.step,
            "state_sha256": self.state_sha256,
            "reason": self.reason,
        }


def _validate_row(payload: object, *, line_number: int) -> ManualCheckpointRejection:
    if not isinstance(payload, dict):
        raise ValueError(f"line {line_number} must contain a JSON object")
    if tuple(payload.keys()) != _EXPECTED_FIELDS:
        if set(payload.keys()) != set(_EXPECTED_FIELDS) or len(payload) != len(_EXPECTED_FIELDS):
            raise ValueError(f"line {line_number} must contain exactly four fields")
        raise ValueError(f"line {line_number} fields must be ordered as {_EXPECTED_FIELDS!r}")

    branch = payload["branch"]
    step = payload["step"]
    state_sha256 = payload["state_sha256"]
    reason = payload["reason"]
    if branch not in _VALID_BRANCHES:
        raise ValueError(f"line {line_number} has an unknown branch")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError(f"line {line_number} step must be a positive integer")
    if not isinstance(state_sha256, str) or _SHA256_PATTERN.fullmatch(state_sha256) is None:
        raise ValueError(f"line {line_number} state_sha256 must be a lowercase SHA-256")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"line {line_number} reason must be a non-blank reason")
    return ManualCheckpointRejection(
        branch=branch,
        step=step,
        state_sha256=state_sha256,
        reason=reason,
    )


def load_manual_checkpoint_rejections(
    path: Path | None,
) -> tuple[ManualCheckpointRejection, ...]:
    if path is None:
        return ()

    rejections: list[ManualCheckpointRejection] = []
    seen_by_identity: set[ManualCheckpointRejection] = set()
    seen_by_branch_step: dict[tuple[str, int], ManualCheckpointRejection] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number} contains malformed JSON") from error
            rejection = _validate_row(payload, line_number=line_number)
            if rejection in seen_by_identity:
                raise ValueError(f"line {line_number} is a duplicate identity")
            previous = seen_by_branch_step.get(rejection.branch_step)
            if previous is not None:
                raise ValueError(f"line {line_number} is a duplicate branch/step")
            seen_by_identity.add(rejection)
            seen_by_branch_step[rejection.branch_step] = rejection
            rejections.append(rejection)
    return tuple(rejections)
