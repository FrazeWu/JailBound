# ASCII-Compatible Projection Vocabulary Design

## Goal

Produce an auditable reviewer example whose optimized `z` and `U` both change
after nearest-neighbor materialization without allowing visibly multilingual or
corrupt token pieces. Continuous optimization must remain unchanged and must
still update only `z` and annotated `U`.

## Scope

Add an optional projection-token policy to the one-sample reviewer runner:

- `special_only`: preserve the current behavior and remain the default.
- `ascii_printable`: exclude special tokens and any vocabulary token whose
  standalone decode is empty, contains U+FFFD, or contains a character outside
  printable ASCII plus `\t`, `\n`, and `\r`.

The policy constrains only nearest-neighbor materialization. It does not alter
the JailBound objective, gradients, Adam state, editable positions, frozen
tokens, checkpoint schedule, response generation, or safety judge.

## Candidate-Set Construction

The runner constructs the allowed token IDs once after loading the tokenizer
and before optimization. Each vocabulary ID is decoded independently with
special-token skipping and cleanup disabled. The `ascii_printable` policy keeps
an ID only when:

1. it is not a tokenizer special ID;
2. its decoded piece is non-empty;
3. its decoded piece does not contain U+FFFD; and
4. every character is printable ASCII or standard ASCII whitespace.

The initial `z` and annotated `U` token IDs must all remain allowed. The runner
fails before optimization if the selected policy excludes an initial editable
token or leaves no projection candidates.

## Data Flow

One immutable allowed-ID set is used for every projection within a run:

1. step-0 projection;
2. each checkpoint projection probe;
3. checkpoint response-pair materialization;
4. final selected-state materialization; and
5. trajectory/report materialization.

Projection remains cosine nearest-neighbor over the allowed embedding rows.
Both `z` and `U` use the same candidate set. Frozen prompt positions continue
to bypass projection entirely.

## CLI And Evidence

Add:

```text
--projection-token-policy {special_only,ascii_printable}
```

The configuration and result artifacts record:

- policy name;
- vocabulary size;
- allowed-token count;
- excluded-token count; and
- SHA-256 of the ordered allowed token IDs.

The recorded optimize command includes the explicit policy. This makes the
qualitative run reproducible and prevents silently presenting constrained
projection as the original unconstrained materializer.

## Acceptance And Failure Handling

Existing acceptance conditions remain strict:

- `z changes > 0` and `U changes > 0`;
- projected `z/U` pass the existing readable-English precheck;
- frozen tokens are unchanged;
- decode/re-tokenize round trip is exact;
- Continuous is judged unsafe;
- Materialized is judged safe; and
- paired evidence comes from the same branch, step, and state hash.

ASCII compatibility is necessary but not sufficient for natural English.
Code-like or otherwise awkward ASCII candidates may pass the machine precheck;
the accepted checkpoint must therefore also receive manual author review before
it is copied into the reviewer-facing example. A manually rejected candidate is
reported honestly and is not relabeled as accepted evidence.

## Backward Compatibility

`special_only` remains the default. Existing configs and materialization APIs
retain their current output unless the new policy is selected explicitly. The
general materialization primitive receives an explicit allowed-ID mechanism so
the runner does not overload `special_token_ids` with non-special exclusions.

## Tests

Tests will establish:

- policy validation and deterministic candidate-set hashing;
- exclusion of special, empty, replacement-character, and non-ASCII pieces;
- retention of printable ASCII words, punctuation, digits, and whitespace;
- rejection when an initial editable token is excluded;
- identical default `special_only` behavior;
- projection selecting only allowed vocabulary rows for both `z` and `U`;
- one immutable candidate-set identity across checkpoint and final paths; and
- CLI/result evidence containing the explicit policy and counts.

## Approved Experiment

After tests pass, rerun sample `jailbound:007843:aa886bf1ef21` on
Qwen2.5-7B-Instruct with the previous fixed seed and hyperparameters,
`learning_rate=0.005`, `projection_token_policy=ascii_printable`, and the same
checkpoint schedule through step 500. Stop at the first checkpoint satisfying
all machine gates. Do not launch a batch run or GBDA baseline.
