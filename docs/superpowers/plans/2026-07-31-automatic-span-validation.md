# Automatic Span Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce frozen span artifacts without hand-edited `accepted` values, retaining an explicit failure ledger.

**Architecture:** The annotator selects semantic spans. Successful `SpanAnnotator` calls have already passed the strict one-span and exact-projection contract and are written as `accepted: true`. Per-row annotation failures are written separately and do not interrupt unrelated rows. The existing freeze procedure accepts the exact boolean rows without an artifact-schema change.

**Tech Stack:** Python, Pydantic, JSONL, pytest.

---

### Task 1: Automatic accepted rows

**Files:**
- Modify: `scripts/calibrate_span_annotation_confidence.py:211-282`
- Test: `tests/safety_eval/test_span_annotation_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
def test_prepare_marks_contract_valid_rows_automatically_accepted() -> None:
    rows = prepare_review_rows({"source": (_raw("source", 0),)}, per_source=1,
                               seed=17, annotator=FixtureAnnotator())
    assert rows[0]["accepted"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/home/dasp/projects/comprehensive_bench/.venv/bin/pytest tests/safety_eval/test_span_annotation_calibration.py::test_prepare_marks_contract_valid_rows_automatically_accepted -q`

Expected: FAIL because the current row has `accepted=None`.

- [ ] **Step 3: Set automatic acceptance after a valid annotation**

```python
row = {
    # Existing immutable prompt, provenance, span, and confidence fields.
    "accepted": True,
}
row["annotation_payload_sha256"] = _annotation_payload_sha256(row)
```

- [ ] **Step 4: Re-run the focused test**

Run: `/home/dasp/projects/comprehensive_bench/.venv/bin/pytest tests/safety_eval/test_span_annotation_calibration.py::test_prepare_marks_contract_valid_rows_automatically_accepted -q`

Expected: PASS.

### Task 2: Recoverable failure ledger

**Files:**
- Modify: `scripts/calibrate_span_annotation_confidence.py:211-282,492-519`
- Test: `tests/safety_eval/test_span_annotation_calibration.py`

- [ ] **Step 1: Write failing continuation coverage**

```python
def test_prepare_records_annotation_failure_and_continues() -> None:
    rows, failures = prepare_automatic_rows(..., annotator=OneFailureAnnotator())
    assert len(rows) == 1
    assert failures[0]["source_row"] == 0
    assert failures[0]["error_type"] == "SpanAnnotationError"
```

- [ ] **Step 2: Run it to verify failure**

Run: `/home/dasp/projects/comprehensive_bench/.venv/bin/pytest tests/safety_eval/test_span_annotation_calibration.py -q`

Expected: FAIL because an exception currently aborts the complete batch.

- [ ] **Step 3: Implement `prepare_automatic_rows` and CLI ledger output**

```python
prepare.add_argument("--failures-output", type=Path, required=True)
# Catch SpanAnnotationError per selected RawExample; emit source, source_row,
# source_row_id, error_type, and error_message into failures. Write valid rows
# to --output and failures atomically to --failures-output.
```

- [ ] **Step 4: Run the focused test suite**

Run: `/home/dasp/projects/comprehensive_bench/.venv/bin/pytest tests/safety_eval/test_span_annotation_calibration.py -q`

Expected: PASS.

### Task 3: Freeze and batch verification

**Files:**
- Test: `tests/safety_eval/test_span_annotation_calibration.py`

- [ ] **Step 1: Add automatic-freeze coverage**

```python
def test_freeze_accepts_automatic_contract_valid_rows(tmp_path) -> None:
    rows = prepare_review_rows(...)
    write_prepared_rows(tmp_path / "accepted.jsonl", rows)
    assert freeze_reviewed_annotations(tmp_path / "accepted.jsonl",
                                       target_precision=1.0,
                                       minimum_selected=1)["accepted_count"] == 1
```

- [ ] **Step 2: Run the regression checks**

Run: `/home/dasp/projects/comprehensive_bench/.venv/bin/pytest tests/safety_eval/test_span_annotation.py tests/safety_eval/test_span_annotation_calibration.py tests/safety_eval/test_v2_manifest_builder.py -q && git diff --check`

Expected: all selected tests pass and the diff is whitespace-clean.

- [ ] **Step 3: Run the automatic batch**

```bash
setsid /home/dasp/projects/comprehensive_bench/.venv/bin/python scripts/calibrate_span_annotation_confidence.py prepare --config configs/benchmark/safety_eval_paper_v2.yaml --per-source 10 --output outputs/calibration/span_annotation_accepted.jsonl --failures-output outputs/calibration/span_annotation_failures.jsonl > outputs/logs/span-annotation-prepare-qwen14b.log 2>&1 < /dev/null &
```
