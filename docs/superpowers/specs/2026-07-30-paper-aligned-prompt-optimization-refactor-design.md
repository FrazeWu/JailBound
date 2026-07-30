# Paper-Aligned Prompt Optimization Refactor

Date: 2026-07-30

Status: Approved in design discussion; awaiting review of this written specification.

## Context

The reviewer evaluation pipeline already models an editable state `e = [z; U]`
and computes FOL jointly over `z` and `U`. However, three behaviors do not match
the JailBound paper:

1. `U` is appended after the complete prompt instead of replacing the seed-intent
   positions `Omega_s`.
2. Materialization appends separately decoded pieces instead of reconstructing the
   prompt by replacing the editable positions.
3. Multi-token answer and refusal anchors are reduced to their first token instead
   of being scored as teacher-forced continuations.

The current JailBound source also does not provide a reliable textual mapping from
its generic `input` intent to the specific harmful content in `output`. Therefore,
the editable positions cannot be recovered faithfully with substring matching or a
fixed suffix length.

This refactor replaces the prompt, objective, and materialization contracts across
all data sources and optimization methods. Existing H1-v2 runs and artifacts remain
immutable legacy evidence and are not interrupted or overwritten.

## Goals

- Make every evaluated prompt pass through an offline de-aligned span annotator.
- Freeze auditable editable character spans in a versioned manifest.
- Map character spans to model-specific token positions without retokenizing prompt
  fragments independently.
- Construct the surrogate input as `[z; Phi_tilde(p; U)]`.
- Evaluate full answer and refusal continuation anchors with teacher forcing.
- Keep exact joint `z + U` FOL and independent dual-branch Adam state.
- Materialize both `z` and `U`, with `z` prepended and `U` replacing its original
  token positions.
- Migrate all continuous and discrete optimization methods to the same prompt and
  scoring interfaces.
- Prevent legacy and paper-aligned artifacts from being mixed in execution or
  aggregation.

## Non-Goals

- Reinterpreting existing schema-v1 checkpoints as paper-aligned checkpoints.
- Inferring editable spans from the last N tokens or an unrecorded heuristic.
- Using the external safety judge during annotation, optimization, selection, or
  materialization.
- Claiming new benchmark results before the corrected pipeline is validated and
  the relevant experiments are rerun.
- Refactoring unrelated taxonomy, model-serving, or report-formatting code.

## Versioning And Compatibility

The corrected contract uses schema version `reviewer_eval.v2`. Schema-v1 manifests,
checkpoints, materializations, and results remain readable only through an explicit
legacy reader. A v2 optimizer or materializer must reject v1 input with a typed
schema error.

V2 output paths include the schema version and a new run identifier. Startup
validation rejects an output root containing v1 artifacts. Aggregation groups by
schema version, transport type, optimization method, and branch, so old and new
results cannot be averaged together.

No active H1-v2 process is terminated. Its locked configuration, manifests, and
outputs remain unchanged.

## Canonical Prompt Contract

### Persistent Annotation

Each source adapter produces the complete prompt text, seed intent, source
provenance, and any source-native structural hints. It does not decide the final
editable span. Every prompt is then sent to the configured de-aligned span
annotator.

The annotator returns one or more exact character spans using a schema equivalent
to:

```text
EditableSpanAnnotation
  start: integer, inclusive
  end: integer, exclusive
  quote: exact substring from the prompt
  role: seed_intent | harmful_payload | attack_instruction
  confidence: float in [0, 1]
  rationale: short diagnostic text
```

The manifest record stores:

```text
StructuredAttackPrompt
  full_text
  seed_intent
  editable_spans
  source and source-row identity
  prompt and intent hashes
  annotator model identity and immutable revision
  annotation prompt-template hash
  raw annotation-response hash
  annotation confidence and validation outcome
```

Character spans are the persistent, tokenizer-independent source of truth. The
manifest must not store an unexplained suffix length as an editable-span substitute.

### Offline Annotation

Annotation occurs during manifest construction, never during optimization. The
annotator runs with a locked model revision, prompt template, decoding settings,
and deterministic seed. Structured output is required.

The deterministic validator checks that:

- at least one span is present;
- `0 <= start < end <= len(full_text)`;
- `full_text[start:end] == quote`;
- spans are ordered and non-overlapping;
- every span confidence meets a threshold frozen by a calibration artifact (the
  record-level confidence is the minimum span confidence);
- the selected text contains at least one non-special token for the surrogate
  tokenizer.

An invalid response receives one schema-repair request containing the validation
errors. If repair fails, the example becomes a typed annotation failure. There is
no suffix, regex, or semantic-similarity fallback.

Source adapters may provide structural hints to the annotator. For example,
HarmBench can identify its context and behavior fields, while SafetyBench can
identify its question and options. The annotator remains the final source of the
editable spans for every source, including apparently trivial direct prompts.

### Tokenized Runtime Form

For a locked tokenizer revision, the full prompt is tokenized once with offset
mapping. A token belongs to `Omega_s` when its character interval overlaps any
validated editable span. If a span cuts through a token, the complete token becomes
editable and the boundary expansion is recorded.

The derived runtime object is equivalent to:

```text
TokenizedEditablePrompt
  base_token_ids
  base_attention_mask
  editable_positions
  frozen_positions
  token_offsets
  span_to_token_mapping
  tokenizer identity and revision
```

Tokenizing editable and frozen fragments separately and concatenating them is
forbidden because it can change BPE boundaries.

## State And Model Input

The continuous state remains:

```text
e = [z; U]
U0 = Embedding(base_token_ids[Omega_s])
```

Soft-prefix initialization is an explicit locked configuration value. The runtime
records the initialization strategy, token IDs or random seed, length, and embedding
hash. There is no implicit initialization default in v2.

To construct `Phi_tilde(p; U)`, the runtime embeds `base_token_ids` and scatters `U`
into `editable_positions`. Frozen positions remain unchanged. The model input is:

```text
H(e) = [z; Phi_tilde(p; U)]
```

This construction is implemented once and shared by objective evaluation,
candidate scoring, diagnostics, embedding-access generation, and all optimizers.

## Continuation-Anchor Objective

Answer and refusal anchors are stored as complete token sequences. For an anchor
`y = [y1, ..., yn]`, its score is:

```text
log P(y | H(e)) = sum_i log P(yi | H(e), y1, ..., y(i-1))
```

The scorer appends the embeddings of `y[:-1]`, applies teacher forcing, and gathers
the target-token log probabilities at the corresponding causal positions. It sums
within each anchor and averages over anchors, exactly matching:

```text
r_tilde(e) = mean_y_in_Yans log P(y | H(e))
             - mean_y_in_Yref log P(y | H(e))
```

All answer and refusal anchors are padded into one scorer batch. The differentiable
scores also provide detached diagnostics, eliminating the current duplicate
diagnostic forward. Anchor length is not normalized because the paper defines a
sum of token log likelihoods for each multi-token anchor.

Configuration loading rejects empty anchors, anchors that tokenize to no ordinary
tokens, and unversioned anchor sets.

## Attack Objective And Search

The objective is:

```text
L_atk(e) = r_tilde(e)
           - gamma_z * ||z - z0||^2
           - gamma_u * ||U - U0||^2

FOL(e) = epsilon * ||grad_(z,U) L_atk(e)||_2
O_minus(e) = L_atk(e) - lambda * FOL(e)
O_plus(e)  = L_atk(e) + lambda * FOL(e)
```

FOL uses one joint norm over both editable blocks and retains the graph required to
differentiate through FOL. Surrogate parameters remain frozen.

The high-value and safety-boundary branches start from separate copies of `e0` and
use independent Adam optimizer state. Every updated state is added to its branch
pool. Configured checkpoints are reporting points, not substitutes for the complete
pools.

Final selection is deterministic and branch-preserving. The main evaluation selects
the top configured `K` states from each pool by descending branch objective, with
ascending step number as the stable tie-breaker. `K` is locked in the run
configuration. A branch with no valid state fails rather than borrowing a state from
the other branch.

Init, random mutation, ZOL, PEZ, GBDA, GCG, O-minus, O-plus, and dual-branch methods
all use the same tokenized prompt and anchor scorer. Their differences are limited
to candidate representation and search/update strategy.

## Materialization

Continuous states project `z` and `U` independently to the surrogate vocabulary.
Discrete methods reuse their selected token IDs. Special or otherwise forbidden
tokens are excluded from projection.

For text-only and API targets:

1. Scatter projected `U` token IDs into `base_token_ids` at `Omega_s`.
2. Assert that every frozen token ID is unchanged.
3. Form `[projected_z_ids; reconstructed_base_ids]`.
4. Decode the complete sequence once with the surrogate tokenizer.
5. Submit the resulting text through the target's standard chat/template interface.

The implementation must not decode `z`, the original prompt, and `U` independently
or join them with inserted spaces.

For targets with a compatible embedding interface, generation consumes
`[z; Phi_tilde(p; U)]` directly. Architecture-specific transformations are allowed
only through a named, versioned adapter. Text materialization and embedding transfer
use distinct transport labels and are not combined in TSR aggregation.

Each materialization record includes:

- schema, run, sample, method, branch, step, and state identity;
- original token IDs, `Omega_s`, projected `z` and `U` token IDs;
- complete reconstructed token IDs and text hash;
- projection cosine diagnostics for both blocks;
- frozen-position invariants and span-boundary expansion;
- full-prompt and editable-span semantic similarity;
- transport type and target adapter identity.

## Evaluation Boundary

The external safety judge is called only after a target model has produced a
response. Judge scores cannot affect span annotation, optimization, pool selection,
projection, semantic filtering, or materialization.

ASR, RCR, and TSR retain their paper definitions. Reports stratify results by schema
version, branch, transport type, source, target, and target family. A result derived
from a v1 checkpoint is always labeled legacy.

## Failure Handling

V2 uses typed failures and does not silently change semantics:

- annotation failure for invalid or low-confidence spans;
- token-mapping failure for unavailable offsets or empty `Omega_s`;
- configuration failure for invalid anchors or unspecified prefix initialization;
- objective failure for non-finite likelihood, gradient, FOL, or branch objective;
- materialization failure for illegal projections or frozen-token mutation;
- compatibility failure when a v1 artifact reaches a v2-only component;
- transport failure when a target embedding adapter is incompatible.

Failures are recorded in ledgers with enough provenance to reproduce them. Resume
logic retries only explicitly retryable execution failures and never changes spans,
tokenizer revisions, or objective configuration.

## Migration Sequence

1. Freeze and inventory all current H1-v2 configuration, manifests, and outputs as
   legacy artifacts.
2. Add v2 prompt, annotation, tokenizer-mapping, and typed-failure contracts.
3. Build the offline span-annotation command and regenerate v2 manifests for every
   configured source.
4. Replace the Transformer input builder and continuation-anchor scorer.
5. Migrate every optimizer to the shared v2 prompt and scorer interfaces.
6. Replace materialization, target generation, judging inputs, and aggregation keys.
7. Update CLI commands and scripts to require an explicit schema and output root.
8. Update `CLAUDE.md`, architecture documentation, and experiment runbooks.
9. Run unit, integration, full-suite, and real-model smoke verification.
10. Switch the default entry point to v2 while retaining explicit legacy readers.

Each stage lands with passing tests before the next stage begins. Expensive full
benchmark reruns occur only after the corrected smoke path is accepted.

## Verification Strategy

### Unit Tests

- Span validation, repair, confidence gating, and typed failures.
- One golden annotated record for every source adapter.
- English, Chinese, repeated text, multiple non-contiguous spans, and partial-token
  boundary expansion.
- Exact scatter reconstruction and frozen-position invariants.
- Hand-calculated multi-token teacher-forced likelihood on a tiny causal LM.
- Joint `z + U` gradients, FOL, second-order differentiation, and Hessian-vector
  products.
- Independent branch optimizer state and per-step pool insertion.
- Continuous and discrete projection with one-pass decode.
- Strict schema-v1/v2 isolation.

### Integration Tests

- Manifest construction from raw source through frozen span annotation.
- Every optimizer using the same `TokenizedEditablePrompt` layout.
- Pool selection through materialization and response-record construction.
- Separate embedding and text transport identities in aggregation.
- Resume behavior preserving annotation and objective identity.

### End-To-End Verification

- Run the complete configured test suite.
- Run a small local surrogate-model optimization with full multi-token anchors.
- Inspect recorded input token order and verify `[z; Phi_tilde(p; U)]`, including
  correct scattering across multiple non-contiguous spans.
- Materialize at least one state from each branch and verify that frozen prompt
  tokens remain identical.
- Generate target responses and judge them only after materialization.
- Confirm that no legacy file is modified and no v1/v2 result is co-aggregated.

## Acceptance Criteria

The refactor is complete when:

- every v2 example has a validated, frozen model-produced annotation or an explicit
  annotation-failure record;
- no v2 execution path uses suffix selection or first-token-only anchors;
- all optimizers share the replacement-based input builder;
- FOL is demonstrably joint over `z` and `U`;
- both blocks are projected and `U` is reconstructed in place;
- frozen token invariants hold in tests and the real-model smoke run;
- old H1-v2 artifacts remain unchanged and separately labeled;
- the full test suite and end-to-end smoke verification pass;
- documentation states the same equations and data flow as the paper and code.
