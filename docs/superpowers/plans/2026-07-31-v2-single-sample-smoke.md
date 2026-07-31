# V2 Single-Sample Smoke Implementation Plan

**Goal:** Execute a mode-isolated one-sample v2 pipeline without weakening the approved benchmark configuration.

### Task 1: Mode-gate the v2 configuration

- [ ] Add `smoke_mode: bool = False` to `V2RunConfig`.
- [ ] Permit a one-source, one-sample, one-method configuration only when `smoke_mode` is true and `output_root` contains `smoke`; keep the existing approved-scope checks otherwise.
- [ ] Add configuration tests for accepted smoke and rejected non-isolated smoke output.

### Task 2: Wire v2 orchestration

- [ ] Dispatch v2 configs to `build_safety_eval_v2_manifests.py`.
- [ ] Read v2 manifests and V2 materialization/response/judgment records in the target phase.
- [ ] Add CLI tests proving v2 dispatch and paths.

### Task 3: Run and verify smoke artifacts

- [ ] Create a local one-sample config with data1 model paths and a smoke output root.
- [ ] Run each stage sequentially and preserve the resulting per-stage artifact paths.
- [ ] Run focused regression tests.
