# V2 Single-Sample Smoke Design

## Goal

Run one real reviewer-eval v2 example through annotation, immutable manifest
construction, optimization, materialization, target generation, both judges,
and aggregation without producing a result that can be mistaken for the
approved 7x17 benchmark.

## Design

Add an explicit `smoke_mode` to the v2 configuration. It permits one source,
one sample, and one configured optimization method only when the output root
contains a `smoke` path component. Every persisted run identity includes the
mode. The standard v2 configuration retains its existing approved-scope
validation.

The CLI will dispatch v2 configurations to the v2 manifest builder and will
use v2 manifest paths and V2 record models through final materialization,
response collection, and judgment counting. Legacy v1 behavior remains on its
existing paths.

## Acceptance

The smoke run has separate JSONL/JSON artifacts for each stage, records both
judge outputs, and the standard CLI refuses to treat smoke artifacts as a
non-smoke benchmark run.
