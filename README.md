# Comprehensive Benchmark Implementation

## Overview

This repository contains the implementation of a modular evaluation and
optimization workflow for studying language-model safety behavior. It includes
dataset adapters, prompt construction, optimization routines, model and judge
interfaces, benchmark runners, and unit tests. The public tree is an
implementation review artifact: it contains code and configuration templates,
but no datasets, trained weights, experimental outputs, figures, or paper
materials.

## Architecture

The code is organized around a small set of independently usable modules:

- `src/generator/` builds structured prompts and prepares data records.
- `src/optimizer/` implements search, candidate selection, and optimization
  utilities.
- `src/models/` and `src/materialization/` provide model-loading adapters.
- `src/judge/` and `src/metrics/` evaluate generated responses and aggregate
  metrics.
- `src/benchmark/` contains baseline, ablation, transfer, and reviewer-facing
  benchmark orchestration.
- `src/defense/`, `src/embedding/`, and `src/objectives/` hold supporting
  training and representation-level components.
- `scripts/` exposes command-line entrypoints for the corresponding workflows.

## Repository Layout

```text
.
├── configs/       # Sanitized configuration templates
├── docs/release/  # Publication scope and configuration notes
├── scripts/       # Workflow entrypoints and data-preparation utilities
├── src/           # Python implementation packages
├── tests/         # Unit and integration test sources
└── pyproject.toml # Package and test metadata
```

## Installation

Use Python 3.11. Create an isolated environment and install the project with
its test dependencies:

```bash
uv sync --extra test
```

Many optional modules require machine-learning libraries and locally available
model weights. Install those dependencies in the environment used for the
specific workflow; they are intentionally not bundled with this artifact.

## Configuration

The templates in `configs/benchmark/` use relative data and output paths.
Code that contacts an OpenAI-compatible service reads these variables when an
endpoint is required:

```bash
export BENCHMARK_API_BASE_URL="http://localhost:8000/v1"
export BENCHMARK_API_KEY="replace-with-local-credential"
```

No endpoint, credential, model directory, or dataset path is supplied by this
repository. See `docs/release/CONFIGURATION.md` for the conventions used by
the retained code.

## Running Checks

The self-contained review-artifact tests do not require model weights or a
network service:

```bash
uv run pytest tests/review_artifact -q
```

Other test modules exercise optional integrations. Run them only after
providing the relevant dependencies, local input files, and configuration.

## Publication Scope

The public repository excludes all local datasets, generated samples,
checkpoints, cached downloads, experiment outputs, logs, figures, plotting
sources, paper files, and archive files. These items may remain in a local
working directory but are ignored by Git. The code therefore documents the
workflow and interfaces without making claims about unavailable experimental
material.

See `docs/release/PUBLICATION_SCOPE.md` for the exact release boundary.

## Reviewer Eval V2

The paper-aligned safety-evaluation path is isolated under
`src/benchmark/safety_eval/`. It consumes only immutable
`reviewer_eval.v2` manifests with offline-annotated editable spans. The
optimized input contract is:

```text
e = [z; U]
H(e) = [z; Phi_tilde(p; U)]
FOL(e) = epsilon * ||grad_(z,U) L_atk(e)||_2
```

`U` replaces only manifest-annotated prompt token positions; all frozen
positions are checked during materialization. Answer and refusal anchors are
scored as complete teacher-forced continuations. Dual JailBound optimization
keeps per-step O-minus/O-plus candidate pools, then deterministically selects
the best final state per branch.

Materialization projects both `z` and `U`, rebuilds the prompt in place, and
decodes the complete token sequence once. Text and embedding transport are
recorded and aggregated separately. Legacy `reviewer_eval.v1` artifacts are
not auto-upgraded and are rejected by v2 execution roots.
