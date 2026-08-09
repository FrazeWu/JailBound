# V2 Token Materialization Integrity

## Scope

V2 evaluation executes only the discrete token sequence produced by
materializing the surrogate optimization state. Continuous embedding transfer
is not a V2 transport because it cannot preserve the same input contract
across independently loaded models.

## Data Contract

Each V2 optimizer state will carry a SHA-256 digest of its saved bytes. The
optimization ledger will persist that digest. A materialization record will
carry both the surrogate tokenizer digest and the optimizer-state digest. A
response record will carry the materialization digest and the exact executed
token-ID digest.

Every resume key includes the relevant immutable digest. Existing rows with a
matching identity but a different digest are a provenance conflict and must
fail explicitly rather than be reused.

## Token Safety And Compatibility

The complete set of tokenizer special IDs is passed into PEZ, GBDA, and GCG
when they are constructed. The same set is used during final projection.

V2 generation requires the target tokenizer digest to equal the materialized
surrogate tokenizer digest. Different-tokenizer target evaluation is rejected
before `generate` receives integer IDs.

## Transport

`TransportType.embedding` remains available for legacy non-V2 helpers, but a
complete V2 materialization must use `TransportType.text`. V2 generation also
defensively rejects another value. This makes the executed path exactly:

`saved z/u embeddings -> nearest permitted surrogate tokens -> complete token IDs -> target generate(input_ids)`.

## Artifact Migration

The smoke runner validates the output root before resuming. Any V2 state or
response record lacking the new digests, or whose response token hash does not
match its materialization, is rejected with a clean-output-root instruction.
It will not silently mix old and new artifacts.

## Tests

Regression tests cover special-ID exclusion for each discrete optimizer,
rejection of V2 embedding transport, target-tokenizer mismatch, state and
materialization digest persistence, digest-aware resume behavior, and old
artifact detection.
