from __future__ import annotations

import pytest

from benchmark.safety_eval.runner import SerialBarrierError, SerialTargetBarrier


def test_first_target_can_start_without_predecessor(tmp_path) -> None:
    barrier = SerialTargetBarrier(tmp_path, ("qwen2_5_7b",))
    barrier.require_ready("qwen2_5_7b")


def test_later_target_requires_predecessor_completion_marker(tmp_path) -> None:
    barrier = SerialTargetBarrier(tmp_path, ("first", "second"))
    with pytest.raises(SerialBarrierError, match="first is incomplete"):
        barrier.require_ready("second")
    barrier.mark_complete("first", response_count=1, primary_judgment_count=1, secondary_judgment_count=1)
    barrier.require_ready("second")


def test_completion_marker_rejects_missing_terminal_stage(tmp_path) -> None:
    barrier = SerialTargetBarrier(tmp_path, ("first",))
    with pytest.raises(SerialBarrierError, match="terminal counts"):
        barrier.mark_complete("first", response_count=1, primary_judgment_count=0, secondary_judgment_count=1)
