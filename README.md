# JailBound

JailBound is an implementation of FOL-guided adversarial prompt optimization
and language-model safety evaluation. It combines continuous prompt search,
semantic-preserving materialization, target-model generation, safety judging,
and aggregate analysis in one reproducible workflow.

## Method

JailBound starts from a structured prompt and optimizes a continuous prompt
state against a differentiable surrogate objective. The workflow records both
the zero-order loss (ZOL) and first-order loss (FOL), then follows two
complementary search directions:

- `jailbound_o_minus` is the high-value branch, which searches toward stronger
  objective values.
- `jailbound_o_plus` is the safety-sensitivity branch, which searches regions
  near the model's local safety boundary.

The optimized state is projected back to discrete text under a calibrated
semantic-preservation constraint. The materialized prompt is then evaluated on
the target model and scored by the configured safety judge. The implementation
also supports same-state continuous-versus-materialized ablations for measuring
intent preservation and materialization fidelity.

## Evaluation Coverage

The focused evaluation uses three complementary sources:

| Source | Role |
| --- | --- |
| HarmBench | Established behavior-level safety benchmark |
| JailBound | Structured prompts organized by risk, domain, and attack type |
| S-Eval | Additional safety-evaluation prompts |

The current comparison workflow includes the initial prompt, random mutation,
ZOL-only optimization, PEZ, GCG, GBDA (including the official-adapter path),
and the JailBound O- and O+ branches. Additional dataset adapters remain in the
repository for compatibility with earlier experiments.

## Repository Layout

```text
.
|-- configs/                         # Experiment configuration files
|-- scripts/                         # Workflow and analysis entrypoints
|-- src/benchmark/safety_eval/       # Evaluation pipeline and optimizers
|-- src/generator/                   # Structured prompt construction
|-- src/materialization/             # Continuous-to-discrete projection
|-- src/judge/ and src/metrics/      # Safety judging and aggregation
|-- tests/                           # Unit and integration tests
`-- additional_experimental_results.md
```

Supporting and legacy packages remain under `src/` for compatibility. The
defense modules are not part of the current JailBound evaluation scope.

## Installation

JailBound requires Python 3.11 and uses `uv` for dependency management:

```bash
uv sync --extra test
```

The Python environment includes the declared machine-learning dependencies.
Model weights, datasets, GPU drivers, and inference services are not bundled
with the repository and must be provided locally.

## Configuration

Experiment configuration files are under `configs/benchmark/`. Before running
an experiment, review the selected file and replace dataset paths, model paths,
output locations, and judge endpoints with values valid for the local system.
OpenAI-compatible services use the following environment variables where
required:

```bash
export BENCHMARK_API_BASE_URL="http://localhost:8000/v1"
export BENCHMARK_API_KEY="replace-with-local-credential"
```

See [Configuration Conventions](docs/release/CONFIGURATION.md) for the retained
configuration rules.

## Quick Start

Run the self-contained checks without model weights or a network service:

```bash
uv run pytest tests/review_artifact -q
```

Validate a configured evaluation and inspect one bounded cell without executing
model inference:

```bash
CONFIG=configs/benchmark/safety_eval_gbda_official.yaml

uv run python scripts/run_safety_eval.py validate --config "$CONFIG"
uv run python scripts/run_safety_eval.py run-smoke \
  --config "$CONFIG" \
  --source harmbench \
  --method jailbound_o_plus \
  --limit 1 \
  --dry-run
```

After local data, models, and judges are configured, a single source-method cell
can be run through optimization, materialization, target evaluation, and
analysis:

```bash
uv run python scripts/run_safety_eval.py optimize \
  --config "$CONFIG" --source harmbench --method jailbound_o_plus

uv run python scripts/run_safety_eval.py materialize \
  --config "$CONFIG" --final-only

uv run python scripts/run_safety_eval.py run-target \
  --config "$CONFIG" --target qwen2_5_7b \
  --source harmbench --method jailbound_o_plus

uv run python scripts/run_safety_eval.py analyze --config "$CONFIG"
```

Use `scripts/run_safety_eval_matrix.py` for resumable source-method matrix
execution. Run each command with `--help` for its complete argument list.

## Results

Aggregate baseline, human-evaluation, behavioral flip-rate, and controlled
evaluation tables are available in
[Additional Experimental Results](additional_experimental_results.md). Raw
model responses and prompt records are intentionally not published.

## Publication Scope

The repository contains implementation code, experiment configurations, tests,
and content-free aggregate result tables. It excludes source datasets, model
weights, generated prompt and response records, checkpoints, runtime logs,
figures, and paper source files. Reproducing the full workflow therefore
requires independently obtained data, models, compute resources, and local
services.

See [Publication Scope](docs/release/PUBLICATION_SCOPE.md) for the detailed
release boundary.

## License

JailBound is released under the [Apache License 2.0](LICENSE).
