# Automatic Span Validation Design

## Goal

Remove the manual `accepted` review step while preserving a deterministic audit
trail for every model-selected editable span.

## Decision

The locked annotation model remains the sole semantic selector. A row is
automatically accepted only when its response satisfies the existing strict
contract: one span, a valid role, and an exact unique projection into the full
prompt. The stored quote and offsets always come from the original prompt.

Model responses that cannot satisfy the contract after the configured repair
attempt are recorded as failures and do not stop unrelated rows. They must not
be silently admitted or converted into an accepted annotation.

No human `accepted` field is required. The automatic artifact records the
contract-validation decision and keeps model confidence for audit, but does not
claim that model confidence is human-calibrated.

## Data Flow

`prepare` will become an automatic preparation command: load examples, ask the
locked annotator, validate the returned span, emit accepted rows for valid
responses, and emit a separate failure ledger for invalid responses. `freeze`
will consume only the automatic accepted rows and create the immutable artifact
without requiring hand-edited booleans. The later manifest builder continues to
use the frozen artifact and retains its existing threshold interface.

## Tests

Tests cover automatic acceptance, rejection recording without batch abort,
missing manual-review fields, and rejection of malformed/ambiguous spans.
