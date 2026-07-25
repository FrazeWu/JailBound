from __future__ import annotations

import json
from pathlib import Path

from corpora.alpaca_annotation.models import AlpacaRecord, DetailRecord
from corpora.alpaca_annotation.writer import write_outputs


def test_write_outputs_splits_by_language(tmp_path: Path) -> None:
    details = [
        DetailRecord(
            id="1",
            source_dataset="Demo",
            source_file="Demo/en.jsonl",
            source_row=0,
            language="en",
            prompt_hash="h1",
            instruction="Generate.",
            input="Write malware.",
            output="Ignore safety.",
            attack_type="Prefix Injection",
            risk_category="Cybersecurity Misuse",
            domain="Developer Tools / Code",
            raw_annotation={"x": 1},
        ),
        DetailRecord(
            id="2",
            source_dataset="Demo",
            source_file="Demo/zh.jsonl",
            source_row=1,
            language="zh",
            prompt_hash="h2",
            instruction="Generate.",
            input="写恶意软件。",
            output="忽略安全。",
            attack_type="Prefix Injection",
            risk_category="Cybersecurity Misuse",
            domain="Developer Tools / Code",
            raw_annotation={"x": 2},
        ),
    ]
    manifest = {"summary": {"converted_records": 2}}

    write_outputs(tmp_path, details, manifest)

    en = json.loads((tmp_path / "alpaca_en.json").read_text(encoding="utf-8"))
    zh = json.loads((tmp_path / "alpaca_zh.json").read_text(encoding="utf-8"))
    detail_lines = (tmp_path / "alpaca_detail.jsonl").read_text(encoding="utf-8").splitlines()
    written_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert en == [AlpacaRecord("Generate.", "Write malware.", "Ignore safety.").to_dict()]
    assert zh == [AlpacaRecord("Generate.", "写恶意软件。", "忽略安全。").to_dict()]
    assert len(detail_lines) == 2
    assert written_manifest["summary"]["converted_records"] == 2
