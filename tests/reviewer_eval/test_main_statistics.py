from __future__ import annotations

import pytest

from benchmark.reviewer_eval.analysis import PairedRow, paired_asr_difference


def test_paired_test_rejects_frozen_aggregate_rows() -> None:
    frozen = [PairedRow(sample_id=None, baseline_unsafe=False, method_unsafe=True, provenance="frozen_pdf")]
    new = [PairedRow(sample_id="sample:1", baseline_unsafe=False, method_unsafe=True, provenance="new_run")]
    with pytest.raises(ValueError, match="frozen aggregates have no paired sample ids"):
        paired_asr_difference(frozen, new)


def test_paired_asr_difference_is_count_first() -> None:
    baseline = [PairedRow("a", False, False), PairedRow("b", True, True)]
    method = [PairedRow("a", False, True), PairedRow("b", True, True)]
    result = paired_asr_difference(baseline, method)
    assert result.delta == .5
    assert result.discordant_method_only == 1
