from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "export_v2_smoke_inspection.py"


def test_exporter_collects_text_ledgers_and_checkpoint_tensors(tmp_path: Path) -> None:
    output_root = tmp_path / "run"
    ledger = output_root / "optimization" / "source" / "method" / "records.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text('{"checkpoint":100,"status":"complete"}\n', encoding="utf-8")
    state = ledger.parent / "states" / "checkpoint.pt"
    state.parent.mkdir(exist_ok=True)
    torch.save({"z": torch.tensor([[1, 2]]), "name": "fixture"}, state)
    destination = output_root / "inspection" / "full_trace.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--destination",
            str(destination),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["jsonl_artifacts"]["optimization/source/method/records.jsonl"] == [
        {"checkpoint": 100, "status": "complete"}
    ]
    checkpoint = payload["checkpoint_states"]["optimization/source/method/states/checkpoint.pt"]
    assert checkpoint["payload"]["z"] == [[1, 2]]
    assert checkpoint["payload"]["name"] == "fixture"
