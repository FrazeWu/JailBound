# GCG-Equivalent ASCII Materialization Design

## Goal

Keep the reviewer experiment's continuous optimization method while making its
checkpoint materialization vocabulary identical to the paper-v2 GCG token
filter. Prevent non-ASCII or control-character responses from qualifying for
early stopping.

## Scope

- Optimize only the existing continuous `z` and `U` blocks.
- Do not replace continuous optimization with discrete GCG.
- Do not add an English dictionary, semantic, grammatical, frequency, or
  common-token filter.
- Do not alter frozen prompt tokens.
- Do not launch batch experiments as part of this change.

## Projection Vocabulary

Enumerate every target-model embedding row. Exclude a token ID when it is a
tokenizer special ID or when decoding that ID alone produces an empty string,
non-ASCII text, or any non-printable character. This is the same predicate used
by `standard_gcg_forbidden_token_ids`.

For every `z` and `U` position, derive a position-specific candidate set from
the global allowed set with that position's initial token ID removed. The
materializer must therefore never report an unchanged token as a valid nearest
neighbor replacement. Persist the global vocabulary counts/hash and the full
position-mask manifest/hash in the result evidence.

The Qwen2.5-7B audit is expected to enumerate 152,064 embedding rows and should
reproduce the observed 61,143 excluded and 90,921 allowed token counts. These
counts are verification evidence, not hard-coded constants.

## Response Qualification

Check both the continuous and materialized target-model responses before any
safety-judge request. A response qualifies only when it is non-empty and every
character is ASCII printable or a normal line separator (`\n` or `\r\n`). Tabs,
other ASCII control characters, replacement characters, and all non-ASCII
characters fail the gate.

A failed response records a branch/checkpoint-specific reason and does not call
the judge or trigger early stopping. Optimization continues to the next
checkpoint. Raw responses remain unchanged in the audit ledger.

## Data Flow

1. Build and hash the GCG-equivalent global ASCII vocabulary.
2. Remove each position's initial token from its local `z` or `U` mask.
3. Optimize continuous `z/U` and project each due checkpoint with those masks.
4. Generate the continuous and materialized responses.
5. Apply the response ASCII/readability gate.
6. Only qualifying pairs reach the safety judge and unsafe-to-safe early-stop
   decision.

## Verification

Tests must first demonstrate the current failures, then cover:

- exact parity between the projection filter and the GCG forbidden-token
  predicate, including exclusion of newline, tab, empty, special, and
  non-ASCII token pieces;
- position-local removal of the initial token ID for both `z` and `U`;
- unchanged frozen prompt IDs and exact decode/re-tokenize behavior;
- non-ASCII/control-character responses rejected before the judge;
- a rejected checkpoint cannot early-stop, while a later valid checkpoint can;
- real Qwen2.5-7B vocabulary counts and manifest hashes recorded from runtime.
