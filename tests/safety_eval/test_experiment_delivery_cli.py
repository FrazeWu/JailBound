from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_experiment_delivery.py"


def _module():
    spec = importlib.util.spec_from_file_location("experiment_delivery", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_delivery_writes_reduced_scope_tables_figure_and_honest_limitations(tmp_path: Path) -> None:
    module = _module()
    matrix = tmp_path / "matrix"
    fol = tmp_path / "fol"
    output = tmp_path / "delivery"
    materialization = tmp_path / "materialization"
    _write_csv(
        matrix / "analysis" / "summary.csv",
        ("Judge", "Target", "Source", "Method", "Threshold", "Unsafe", "N", "ASR"),
        [{"Judge": "primary", "Target": "qwen", "Source": "s_eval", "Method": "init", "Threshold": 0.5, "Unsafe": 1, "N": 17, "ASR": "1 / 17 (5.88%)"}],
    )
    _write_csv(
        matrix / "analysis" / "paired_vs_init.csv",
        ("Judge", "Target", "Source", "Method", "Threshold", "N", "Net ASR change", "Method-only", "Init-only", "McNemar p"),
        [{"Judge": "primary", "Target": "qwen", "Source": "s_eval", "Method": "gcg", "Threshold": 0.5, "N": 17, "Net ASR change": "1 / 17", "Method-only": 1, "Init-only": 0, "McNemar p": "1"}],
    )
    _write_csv(
        matrix / "analysis" / "threshold_ranks.csv",
        ("Judge", "Target", "Source", "Threshold", "Method", "Rank"),
        [{"Judge": "primary", "Target": "qwen", "Source": "s_eval", "Threshold": 0.5, "Method": "init", "Rank": 1}],
    )
    _write_csv(
        fol / "analysis" / "fol_bfr.csv",
        ("Source", "Band", "Radius", "Judge", "Prompts", "Eligible prompts", "Sparse prompts", "Accepted directions", "Mean BFR", "Provenance"),
        [{"Source": "s_eval", "Band": "high", "Radius": 0.1, "Judge": "primary", "Prompts": 7, "Eligible prompts": 7, "Sparse prompts": 0, "Accepted directions": 56, "Mean BFR": 0.0, "Provenance": "new_run"}],
    )
    _write_csv(
        fol / "analysis" / "fol_d50.csv",
        ("Source", "Sample ID", "Band", "Judge", "Usable radii", "d50", "Lower bound", "Right censored", "Provenance"),
        [{"Source": "s_eval", "Sample ID": "opaque-id", "Band": "high", "Judge": "primary", "Usable radii": 4, "d50": "", "Lower bound": 0.4, "Right censored": True, "Provenance": "new_run"}],
    )
    _write_csv(
        fol / "analysis" / "fol_h4_controls.csv",
        ("Scope", "Held-out rows", "Controls AUROC", "Controls-plus-FOL AUROC", "Delta AUROC", "Controls AUPRC", "Controls-plus-FOL AUPRC", "Delta AUPRC", "Controls Brier", "Controls-plus-FOL Brier", "Delta Brier", "Controls ECE", "Controls-plus-FOL ECE", "Delta ECE", "Margin threshold", "Status", "Provenance"),
        [{"Scope": "combined", "Held-out rows": "", "Controls AUROC": "", "Controls-plus-FOL AUROC": "", "Delta AUROC": "", "Controls AUPRC": "", "Controls-plus-FOL AUPRC": "", "Delta AUPRC": "", "Controls Brier": "", "Controls-plus-FOL Brier": "", "Delta Brier": "", "Controls ECE": "", "Controls-plus-FOL ECE": "", "Delta ECE": "", "Margin threshold": "", "Status": "inconclusive", "Provenance": "new_run"}],
    )
    (fol / "analysis" / "fol_boundary_claim.json").write_text(json.dumps({"decision": "inconclusive", "reason": "interpolation_underpowered"}), encoding="utf-8")
    init_ledger = materialization / "optimization" / "s_eval" / "init" / "materialization.jsonl"
    init_ledger.parent.mkdir(parents=True)
    init_ledger.write_text(json.dumps({"source": "s_eval", "method": "init", "checkpoint": 0, "status": "complete", "intent_preserved": True, "semantic_similarity_after": 0.9, "prefix_projection_cosine": 0.8, "seed_projection_cosine": 0.7}) + "\n", encoding="utf-8")
    excluded_ledger = materialization / "optimization" / "s_eval" / "random_mutation" / "materialization.jsonl"
    excluded_ledger.parent.mkdir(parents=True)
    excluded_ledger.write_text(json.dumps({"source": "s_eval", "method": "random_mutation", "checkpoint": 100, "status": "complete", "intent_preserved": True, "semantic_similarity_after": 0.1, "prefix_projection_cosine": 0.2, "seed_projection_cosine": 0.3}) + "\n", encoding="utf-8")

    manifest = module.build_delivery(matrix_root=matrix, fol_root=fol, output_root=output, base_materialization_root=materialization)

    assert manifest["scope"] == {"sources": ["s_eval"], "samples_per_source": 17}
    assert (output / "tables" / "fair_optimization.csv").is_file()
    assert (output / "tables" / "paired_vs_init.csv").is_file()
    assert (output / "tables" / "judge_threshold_sensitivity.csv").is_file()
    assert (output / "tables" / "fol_boundary_diagnostics.csv").is_file()
    fidelity = (output / "tables" / "materialization_fidelity.csv").read_text(encoding="utf-8")
    assert "random_mutation" not in fidelity
    assert (output / "figures" / "fol_boundary_diagnostics.pdf").is_file()
    assert (output / "figures" / "fol_boundary_diagnostics.png").is_file()
    assert "inconclusive" in (output / "analysis" / "limitations.md").read_text(encoding="utf-8")
    assert "not evaluated" in (output / "analysis" / "evaluation_coverage.md").read_text(encoding="utf-8")


def test_build_delivery_rejects_a_missing_fol_claim(tmp_path: Path) -> None:
    module = _module()
    matrix = tmp_path / "matrix"
    fol = tmp_path / "fol"
    _write_csv(
        matrix / "analysis" / "summary.csv",
        ("Judge", "Target", "Source", "Method", "Threshold", "Unsafe", "N", "ASR"),
        [{"Judge": "primary", "Target": "qwen", "Source": "s_eval", "Method": "init", "Threshold": 0.5, "Unsafe": 1, "N": 17, "ASR": "1 / 17 (5.88%)"}],
    )

    try:
        module.build_delivery(matrix_root=matrix, fol_root=fol, output_root=tmp_path / "delivery")
    except ValueError as error:
        assert "FOL claim" in str(error)
    else:
        raise AssertionError("missing FOL claim must fail closed")
