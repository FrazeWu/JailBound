from __future__ import annotations

from pathlib import Path

from benchmark.reviewer_eval.registry import FrozenRegistry
from benchmark.reviewer_eval.schema import CellKey


ROOT = Path(__file__).resolve().parents[2]
PDF = Path("/home/dasp/.codex/attachments/00fe878e-dfb2-40ec-97fd-49c9f3da27cf/3527_JailBound_A_FOL_Guided_Ja.pdf")


def _requested() -> CellKey:
    return CellKey(dataset_source="advbench", sample_manifest_hash="a" * 64, optimization_method="pez",
                   optimization_budget="updates=100", surrogate_model_revision="qwen", target_model_revision="qwen",
                   decoding_config_hash="b" * 64, judge_revision="octopus", judge_threshold=.5)


def test_unknown_pdf_identity_never_skips_new_cell() -> None:
    registry = FrozenRegistry.load(ROOT / "configs/benchmark/reviewer_frozen_pdf.yaml", PDF)
    assert registry.find_exact(_requested()) is None


def test_frozen_rows_keep_pdf_locator() -> None:
    registry = FrozenRegistry.load(ROOT / "configs/benchmark/reviewer_frozen_pdf.yaml", PDF)
    row = registry.context_rows(table=2, row="AdvBench", column="High-value")[0]
    assert row.pdf_page == 7
    assert row.value == 61.9
    assert row.provenance == "frozen_pdf"
