from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.safety_eval.replacement_parent import (
    assemble_replacement_parent,
    seed_reused_secondary_judgments,
)


def _source_artifacts(root: Path, source: str) -> None:
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    (root / "manifests" / f"controlled_{source}.jsonl").write_text("{}\n", encoding="utf-8")
    for path in (
        root / "optimization" / source,
        root / "responses" / "qwen2_5_7b" / source,
        root / "judgments" / "octopus_seval_14b" / "qwen2_5_7b" / source,
    ):
        path.mkdir(parents=True, exist_ok=True)
        (path / "records.jsonl").write_text("{}\n", encoding="utf-8")


def test_assemble_replacement_parent_links_only_declared_source_artifacts(tmp_path: Path) -> None:
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    for source in ("harmbench", "s_eval"):
        _source_artifacts(original, source)
    _source_artifacts(replacement, "jailbound")

    parent = assemble_replacement_parent(
        original_root=original,
        replacement_root=replacement,
        output_root=tmp_path / "combined",
        target_key="qwen2_5_7b",
        primary_judge="octopus_seval_14b",
    )

    assert (parent / "manifests" / "controlled_harmbench.jsonl").is_symlink()
    assert (parent / "manifests" / "controlled_s_eval.jsonl").is_symlink()
    assert (parent / "manifests" / "controlled_jailbound.jsonl").is_symlink()
    assert (parent / "optimization" / "jailbound").resolve().is_relative_to(replacement)
    assert (parent / "responses" / "qwen2_5_7b" / "harmbench").resolve().is_relative_to(original)
    provenance = json.loads((parent / "replacement_provenance.json").read_text(encoding="utf-8"))
    assert provenance["replaced_source"] == "jailbound"
    assert provenance["sources"]["jailbound"] == str(replacement.resolve())


def test_assemble_replacement_parent_rejects_missing_required_artifacts(tmp_path: Path) -> None:
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    _source_artifacts(original, "harmbench")
    _source_artifacts(original, "s_eval")
    replacement.mkdir()

    with pytest.raises(ValueError, match="required artifact is missing"):
        assemble_replacement_parent(
            original_root=original,
            replacement_root=replacement,
            output_root=tmp_path / "combined",
            target_key="qwen2_5_7b",
            primary_judge="octopus_seval_14b",
        )


def test_seed_reused_secondary_judgments_links_only_unchanged_sources(tmp_path: Path) -> None:
    original = tmp_path / "original-selection"
    for source in ("harmbench", "s_eval"):
        path = original / "judgments" / "qwen32_compat" / "qwen2_5_7b" / source
        path.mkdir(parents=True)
        (path / "records.jsonl").write_text("{}\n", encoding="utf-8")

    selection = seed_reused_secondary_judgments(
        original_selection_root=original,
        output_root=tmp_path / "replacement-selection",
        target_key="qwen2_5_7b",
    )

    assert (selection / "judgments" / "qwen32_compat" / "qwen2_5_7b" / "harmbench").is_symlink()
    assert (selection / "judgments" / "qwen32_compat" / "qwen2_5_7b" / "s_eval").is_symlink()
    assert not (selection / "judgments" / "qwen32_compat" / "qwen2_5_7b" / "jailbound").exists()
