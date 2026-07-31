from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_interpolation_generation_reads_only_requested_source_materializations(tmp_path: Path) -> None:
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_fol_interpolation_generation_resume.py"
    spec = importlib.util.spec_from_file_location("fol_interpolation_generation_resume", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    records = tmp_path / "interpolation_materialization" / "s_eval" / "fol_interpolation"
    records.mkdir(parents=True)
    payload = {"source": "s_eval", "method": "fol_interpolation", "sample_id": "opaque-id"}
    (records / "records.jsonl").write_text(json.dumps(payload) + "\n", encoding="utf-8")

    assert module.interpolation_materializations(tmp_path, "s_eval") == [payload]
