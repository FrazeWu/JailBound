# Position-Aware Common-English Projection Design

## Goal

Produce one reviewer-facing English materialization-fidelity example for sample
`jailbound:007843:aa886bf1ef21` whose optimized `z` and annotated `U` both
change, while preventing nearest-neighbor projection from producing code
identifiers, mixed-language fragments, malformed token joins, or new proper
nouns. Continuous optimization remains unchanged and continues to update only
`z` and annotated `U`.

## Scope

Add an optional projection-token policy to the one-sample reviewer runner:

- `special_only`: preserve the unconstrained default behavior.
- `ascii_printable`: preserve the existing printable-ASCII global mask.
- `english_common_positioned`: apply a deterministic candidate mask chosen
  from the initial token's lexical position class.

The new policy changes only nearest-neighbor materialization. It does not
change the JailBound objective, gradients, Adam state, editable positions,
frozen prompt tokens, checkpoint schedule, target generation, or safety judge.
No GBDA or batch experiment is included.

## Deterministic English Resource

Pin `wordfreq==3.1.1` as a project dependency and use
`wordfreq.zipf_frequency(word, "en")` with a fixed minimum Zipf frequency of
`3.5`. The dependency version, language, and threshold are part of the result
configuration.

The policy never downloads data during a run. Candidate construction fails
before optimization if the pinned package is unavailable or reports a version
different from the configured version.

## Position Classes

Each initial editable token is decoded independently with special-token
skipping and cleanup disabled, then assigned exactly one class:

1. `word_start`: exactly one leading ASCII space followed by one or more
   lowercase ASCII letters.
2. `sentence_initial`: no leading whitespace, beginning with an uppercase ASCII
   letter, followed only by lowercase ASCII letters.
3. `continuation`: no leading whitespace and composed only of lowercase ASCII
   letters.
4. `contraction`: an apostrophe followed only by lowercase ASCII letters.
5. `punctuation`: ASCII punctuation with no letters or digits.
6. `other`: every remaining piece.

Only `word_start` positions may introduce a new token ID. Every other class is
restricted to the original token ID at that position. This preserves the
initial sentence boundary, token joins, suffix structure, punctuation, and
contractions while still allowing changes at complete-word positions in both
`z` and `U`.

## Common-English Candidate Set

A vocabulary token is a replaceable `word_start` candidate only when all of
the following hold:

1. it is not a tokenizer special ID;
2. its token ID is below the exclusive rank ceiling `50_000`;
3. its standalone decoded piece matches ` [a-z]+` exactly;
4. its stripped word has English Zipf frequency at least `3.5`; and
5. it contains no digits, underscores, uppercase letters, control characters,
   non-ASCII characters, or punctuation.

The original token ID is always added to its own position mask, even when it is
outside the common-English set. This is the only exception and exists to keep
step 0 exactly reconstructible. Original IDs are not added to other positions'
masks.

The rank ceiling is a secondary protection against rare tokenizer artifacts;
the English frequency test is the primary lexical filter. The fixed prompt may
still contain proper nouns such as `China`, but optimization cannot introduce a
new proper noun into `z` or `U`.

## Materialization API And Data Flow

Extend the materialization primitive with optional position-specific allowed
token IDs for `z` and `U`. Existing global `allowed_token_ids` behavior remains
unchanged and mutually exclusive with position-specific masks.

For `english_common_positioned`, the runner constructs all masks once after
loading the tokenizer and before optimization. The same immutable masks are
then reused for:

1. step-0 projection;
2. every checkpoint projection probe;
3. checkpoint continuous/materialized response generation;
4. trajectory serialization;
5. selected-state materialization; and
6. final result/report serialization.

Nearest-neighbor selection remains cosine similarity over the vocabulary rows
permitted at each position. Frozen prompt positions continue to bypass
projection.

## CLI And Evidence

Extend the existing option to:

```text
--projection-token-policy {special_only,ascii_printable,english_common_positioned}
```

The result configuration records:

- policy name;
- `wordfreq` package version;
- language code and Zipf threshold;
- exclusive token-ID ceiling;
- vocabulary size;
- common-English candidate count and ordered-ID SHA-256;
- position class for every `z` and `U` token;
- allowed-token count and ordered-ID SHA-256 for every position; and
- a SHA-256 over the complete ordered position-mask manifest.

The optimize command records the explicit policy. This distinguishes the
constrained reviewer run from the default materializer and makes the evidence
auditable.

## Acceptance And Manual Review

The existing machine gates remain mandatory:

- projected `z changes > 0` and `U changes > 0`;
- frozen prompt tokens remain unchanged;
- one-pass decode followed by re-tokenization is exact;
- Continuous is judged unsafe;
- Materialized is judged safe; and
- both responses share branch, step, and continuous-state SHA-256.

Before response generation or judge calls, the runner also verifies that every
projected ID belongs to its recorded position mask. A mismatch fails the run
rather than silently widening the candidate set.

Machine acceptance is not reviewer acceptance. The first machine-accepted
checkpoint is manually checked for grammatical, readable English and faithful
`z/U` labeling. Code-like, semantically incoherent, or visibly unnatural text
is rejected and is not copied into a reviewer-facing compact file.

An optional manual-rejection ledger records each rejected candidate's branch,
step, continuous-state SHA-256, and concise reason. On a deterministic replay,
the runner verifies that an encountered candidate has the same identity as the
ledger entry, records `manually_rejected` in its checkpoint decision, and
continues to the next declared checkpoint. A ledger entry that is not
encountered with the exact state hash fails closed. The ledger path and file
SHA-256 are included in the run configuration.

## Failure Handling

Candidate construction fails before GPU optimization when:

- the policy name is unknown;
- the English-frequency dependency or version is unavailable;
- no common-English replacement candidates remain;
- a position mask is empty;
- a position mask omits its original token ID; or
- a stored mask hash does not match the mask used by a materialization path.

The experiment pauses at the first machine-qualified checkpoint for manual
review, up to step 500. If the candidate is rejected, its identity is appended
to the manual-rejection ledger and the deterministic search is replayed with
that candidate excluded so later checkpoints can be examined. If all
checkpoints are exhausted, the result remains `UNVERIFIED` and no compact
success file is created.

## Backward Compatibility

`special_only` remains the default. Existing call sites, configs, and outputs
retain global-mask behavior unless `english_common_positioned` is explicitly
selected. `ascii_printable` remains available for reproducing the preceding
run.

## Tests

Tests establish:

- deterministic class assignment for all six position classes;
- exclusion of special tokens, non-ASCII text, code identifiers, CamelCase,
  digits, punctuation, rare words, and token IDs at or above `50_000` from the
  common-English replacement set;
- retention of frequent lowercase English word-start pieces;
- per-position inclusion of the original ID and exclusion of another
  position's exceptional original ID;
- unchanged global behavior for `special_only` and `ascii_printable`;
- validation of empty, malformed, or mutually conflicting masks;
- nearest-neighbor selection from the correct position mask for both `z` and
  `U`;
- identical mask-manifest hash across checkpoint, trajectory, selected-state,
  and final materialization paths;
- CLI and result evidence containing the explicit policy and lexical metadata;
- fail-closed validation and deterministic skipping of an exact
  branch/step/state-hash manual rejection, while an altered or missing identity
  is rejected; and
- truthful judge-use reporting in early-stop and non-early-stop paths.

## Approved Experiment

After implementation tests and independent review pass, rerun
`jailbound:007843:aa886bf1ef21` on Qwen2.5-7B-Instruct with the fixed seed
`20260725`, learning rate `0.005`, the existing optimization coefficients,
`english_common_positioned`, and the declared checkpoint schedule through step
500. Inspect every machine-accepted checkpoint manually. Create the compact
reviewer example only after all machine and manual gates pass. When a candidate
fails manual review, record it in the rejection ledger and continue until a
later checkpoint passes or step 500 is exhausted.
