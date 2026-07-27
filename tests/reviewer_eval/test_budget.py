from __future__ import annotations

import pytest

from benchmark.reviewer_eval.optimizers.base import BudgetExceeded, BudgetLedger, CheckpointEmitter


def test_dual_budget_is_exactly_fifty_fifty() -> None:
    ledger = BudgetLedger(update_limit=100, candidate_limit=3200, branch_limits={"o_minus": 50, "o_plus": 50})
    for index in range(100):
        ledger.consume_update("o_minus" if index % 2 == 0 else "o_plus")
    assert ledger.updates == 100
    assert ledger.branch_updates == {"o_minus": 50, "o_plus": 50}
    with pytest.raises(BudgetExceeded):
        ledger.consume_update("o_minus")


def test_checkpoint_emitter_emits_once_in_order() -> None:
    emitter = CheckpointEmitter([0, 25, 50, 100])
    assert [emitter.due(step) for step in [0, 1, 25, 25, 50, 99, 100]] == [True, False, True, False, True, False, True]


def test_ledger_records_computation_counters_safely() -> None:
    ledger = BudgetLedger(update_limit=1, candidate_limit=1)

    ledger.record_forward()
    ledger.record_backward(2)
    ledger.record_hvp()

    assert (ledger.forward_passes, ledger.backward_passes, ledger.hvp_calls) == (1, 2, 1)
    with pytest.raises(ValueError, match="positive"):
        ledger.record_forward(0)
