from __future__ import annotations

from benchmark.safety_eval.metrics import Outcome, Rate, compute_asr


def test_rate_preserves_count_and_two_decimal_display() -> None:
    rate = Rate.from_flags([True] * 7 + [False] * 4)
    assert (rate.numerator, rate.denominator) == (7, 11)
    assert rate.display == "7 / 11 (63.64%)"


def test_failed_examples_remain_in_intention_to_evaluate_denominator() -> None:
    outcomes = [Outcome.complete(True), Outcome.complete(False), Outcome.failed("generation")]
    assert compute_asr(outcomes, execution_only=False).display == "1 / 3 (33.33%)"
    assert compute_asr(outcomes, execution_only=True).display == "1 / 2 (50.00%)"
