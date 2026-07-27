# Reviewer-Focused Evaluation Additions

Date: 2026-07-25

## Material Passport

- Origin Skill: brainstorming + experiment-agent
- Origin Mode: experiment plan
- Origin Date: 2026-07-25
- Verification Status: DESIGN APPROVED; RESULTS UNVERIFIED
- Version Label: reviewer_experiment_design_v1
- Primary manuscript: `3527_JailBound_A_FOL_Guided_Ja.pdf`
- Manuscript SHA-256:
  `f44b5bf816d333218f109ca0af216b6635dd0921875f56eaaa488bf6b7cf2b91`
- Reviewer summary: `comments.md`

## Objective

Add the smallest defensible set of evaluation-side experiments needed to
address the reviewers' main empirical concerns:

1. compare all attack sources under the same optimization budget;
2. add embedding/discrete optimization baselines and a no-FOL ablation;
3. add S-Eval to the data layer;
4. measure materialization fidelity and semantic drift;
5. reduce dependence on one judge and report uncertainty;
6. test whether FOL is an empirical proxy for proximity to a safety decision
   boundary rather than merely a high-loss, high-curvature, or noisy signal.

The intended boundary claim is empirical and deliberately limited:

> FOL serves as an empirical proxy for proximity to safety decision
> boundaries when high-FOL prompts show greater local safety-label
> instability, smaller measured flip distance, and FOL concentration near
> independently observed behavior/margin crossings.

The experiments must not claim that FOL is mathematically equal to a safety
boundary.

## Confirmed Execution Overrides

The following user-confirmed execution decisions supersede conflicting sample
counts, model identities, judge names, and resource assumptions in the
historical design text below:

- The only main-matrix surrogate and target is the local
  `Qwen2.5-7B-Instruct` snapshot; no additional model is downloaded.
- The main matrix uses 17 controlled examples from each of the seven sources.
- The focused FOL study remains restricted to JailBound and S-Eval, with 17
  validation examples per source split into 7 low-FOL, 3 middle-FOL, and 7
  high-FOL examples. Five disjoint radius-calibration examples remain reserved
  per source.
- The focused study retains 21 interpolation points and at least 17
  semantically valid positions per path. With seven matched low/high pairs,
  at least five valid paths are required for a non-underpowered interpolation
  analysis.
- The primary judge is local Octopus-SEval-14B. The secondary judge is the
  local Qwen3-32B-AWQ OpenAI-compatible endpoint. No defense, fine-tuning, or
  repair experiment is in scope.

The locked configuration and run manifests are the authoritative source for
the exact executed identities.

## Scope

### Included

- Attack-prompt optimization and target-model safety evaluation.
- Seven fixed benchmark sources.
- A shared white-box surrogate and four serially evaluated target models.
- Optimization baselines, direct FOL ablations, materialization diagnostics,
  judge robustness, uncertainty, and the focused FOL-boundary experiment.

### Excluded

- Defense, safety fine-tuning, data relabeling, or iterative repair.
- Training an attack-prompt generator or regenerating the benchmark taxonomy.
- Re-running an exact result cell already reported in the submitted PDF or
  replacing its frozen value.
- Formal human evaluation. The author audit is an error analysis only.
- Claims of causal equivalence between gradient norm and a safety boundary.

## Result Provenance and Freeze Policy

The submitted PDF is the only source of frozen results. Repository LaTeX files
whose captions or comments identify values as placeholders are not evidence and
must not populate the result registry.

Every result cell has the identity:

```text
dataset_source
+ sample_manifest_hash
+ optimization_method
+ optimization_budget
+ surrogate_model_revision
+ target_model_revision
+ decoding_config
+ judge_revision
+ judge_threshold
```

A PDF value is marked `frozen_pdf` and skipped only when all fields that can be
established from the manuscript match the requested cell. A new controlled
subset is a different experiment and is marked `new_run`; it does not replace a
PDF cell even when the dataset, method, and model names match.

Frozen aggregate values may be reported as historical context. Because their
per-example predictions are unavailable, they must not be included in paired
tests, newly computed confidence intervals, or claims based on a shared sample.
The provenance registry must preserve the PDF table number, row, column, and
page locator for every frozen value.

## Fixed Data Layer

The benchmark sources are:

1. AdvBench
2. HarmBench
3. SafetyBench
4. SG-Bench
5. JailbreakBench
6. JailBound
7. S-Eval (`S-Eval_attack_en_full.jsonl`)

The main fair-comparison experiment uses 50 examples per source. Selection is
deterministic and stratified by the source labels after mapping them into the
JailBound risk-category and attack-type inventories. JailBound sampling
preserves its hierarchical threat conditions; S-Eval sampling preserves its
risk taxonomy and attack-template distribution.

The global selection seed is `20260725`. Method-level and perturbation-level
seeds are derived deterministically from this seed and the stable cell id.

The manifest is immutable after optimization begins. Each record contains:

- stable example id and source row;
- source dataset and source file hash;
- original intent and original attack text;
- canonical risk category, threat domain, and attack type;
- selection stratum and selection seed;
- prompt hash and manifest hash;
- language and preprocessing decisions.

No method may skip a difficult example silently. Unsupported or failed examples
remain in the denominator and are recorded with an explicit failure reason,
unless the failure is proven to be a source-data error before any method runs.

## Models and Execution Order

White-box optimization is performed once on:

- `Qwen/Qwen2.5-14B-Instruct`

Materialized prompts are then evaluated serially on:

1. `Qwen/Qwen2.5-14B-Instruct`
2. `Qwen/Qwen2.5-7B-Instruct`
3. `meta-llama/Llama-3.1-8B-Instruct`
4. `google/gemma-2-9b-it`

All datasets and methods for one target model are completed before loading the
next target model. Two GPUs may hold identical replicas of a 7B-9B target for
sample sharding, but two different target models must not run concurrently.
Model repository ids, resolved revisions, tokenizer revisions, chat templates,
dtype, and generation settings are written to every result manifest.

Target generation uses greedy decoding, temperature 0, and a maximum of 512
new tokens unless a model interface requires a documented compatibility
override.

## Optimization Methods

The controlled comparison includes:

| Method | Definition |
|---|---|
| Init | Unoptimized source prompt. |
| Random Mutation | Random semantic-preserving mutation with the shared candidate budget. |
| ZOL-only / no-FOL | Optimize `L_atk` with `lambda = 0` using the JailBound parameterization. |
| PEZ | Continuous prompt optimization followed by hard token projection. |
| GBDA | Gumbel-Softmax optimization over editable token distributions. |
| GCG | Discrete coordinate-gradient token replacement. |
| JailBound O- | `L_atk - lambda * FOL`, the high-value branch. |
| JailBound O+ | `L_atk + lambda * FOL`, the safety-boundary branch. |
| Dual-branch | Equal-compute combination of O- and O+ candidates. |

All optimizing methods receive 100 updates or candidate evaluations per
example. Dual-branch receives 50 O- updates and 50 O+ updates, not twice the
budget. Checkpoints are recorded at 0, 25, 50, and 100 total updates. Init has
zero optimization cost and is evaluated on the same manifest.

Methods share:

- editable token positions and maximum editable-token count;
- soft-prefix length where the method supports a continuous prefix;
- answer/refusal anchors and attack objective;
- semantic-intent acceptance policy;
- maximum candidate count and checkpoint policy;
- materialization and target-generation settings.

Method-specific representations remain intact: GCG is not forced into a
continuous parameterization, and PEZ/GBDA are not credited with extra candidate
evaluations hidden inside projection. Forward passes, backward passes,
second-order products, generated candidates, and accepted candidates are
recorded so compute differences remain visible.

## Semantic Preservation and Materialization

The immutable malicious seed intent, not the full jailbreak wrapper, is the
semantic reference. A candidate is accepted only if:

1. its materialized text remains in the same risk category and attack intent;
2. its multilingual embedding similarity to the seed intent is at least
   `tau_sem`;
3. it does not lose or invert the target behavior through truncation;
4. it is executable text under the target tokenizer.

`tau_sem` is calibrated before the experiment on held-out paraphrase and
non-paraphrase pairs by selecting the threshold that retains 95% of accepted
intent-preserving paraphrases. Semantic similarity uses `BAAI/bge-m3`; its
resolved model revision is pinned in the run manifest. The calibration examples
are never used in the evaluation manifest. The threshold, encoder revision,
and calibration summary are frozen in the run manifest.

Materialization reporting includes:

- mean and distribution of embedding-to-token projection cosine similarity;
- seed-intent semantic similarity before and after materialization;
- intent-preservation rate and rejection rate;
- attack-score and ASR change before versus after materialization where both
  interfaces are executable;
- representative success, semantic-drift, and projection-failure examples.

## Main Evaluation Metrics

Primary metrics:

- attack success rate (ASR);
- ASR change from Init on the same examples;
- continuous judge score;
- risk coverage rate (RCR);
- transfer success rate (TSR) from the fixed surrogate;
- semantic-intent preservation rate.

Efficiency metrics:

- wall-clock seconds per example;
- peak allocated GPU memory;
- forward, backward, and Hessian-vector-product counts;
- attempted and accepted candidate counts;
- total generated and judged tokens.

Each percentage is reported as `success_count / denominator (xx.xx%)`. A single
shared decimal-rounding function is used for all tables. The count is the
authoritative value; the two-decimal percentage is display-only.

## Judge Robustness and Author Audit

Octopus-SEval is the primary judge. WildGuard is the secondary judge. Both
judge revisions, prompts, label mappings, score normalization, and thresholds
are frozen before evaluating method identities.

Report:

- ASR and confidence intervals under each judge;
- inter-judge agreement and disagreement rate;
- ASR at normalized unsafe thresholds
  `max(0, eta - 0.1)`, `eta`, and `min(1, eta + 0.1)`;
- method rankings under each judge and threshold;
- a stratified author audit of approximately 100 responses, prioritizing judge
  disagreements, semantic-filter edge cases, and apparent false positives or
  false negatives.

The audit is labeled `author error analysis`, not `human evaluation`. It cannot
be used to claim population-level human agreement.

## FOL Boundary-Proximity Validation

### Research Question

Does FOL identify prompts closer to an independently observed safety decision
boundary, or does it merely identify regions with high attack loss, curvature,
or local noise?

### Operational Boundary Definitions

The experiment uses two non-identical definitions:

1. **Behavior boundary**: the smallest semantic-preserving local perturbation
   at which Qwen2.5-14B changes between refusal/safe behavior and unsafe
   compliance under an external judge.
2. **Internal margin boundary**: the point where a calibrated difference
   between answer-oriented and refusal-oriented continuation scores reaches
   the behavior-calibrated threshold `m_star`.

FOL is not used to define either boundary.

### Hypotheses

- H1: At a fixed perturbation radius, high-FOL prompts have a higher behavior
  flip rate than matched low-FOL prompts.
- H2: FOL is negatively associated with estimated behavior and internal-margin
  boundary distance.
- H3: Along semantically valid safe-unsafe interpolation paths, the FOL peak is
  closer to the observed crossing than a random peak location and a curvature
  peak.
- H4: Adding FOL improves boundary-flip prediction beyond attack loss, prompt
  length, internal margin, curvature, and local roughness.

### Validation Sample

The validation sample is selected from the controlled manifests:

- JailBound: 31 prompts.
- S-Eval: 31 prompts.
- Total: 62 prompts.

For each source, prompts are ranked by FOL after preserving risk-category and
initial-label coverage:

- lower-FOL reported group: 11 prompts;
- middle band: 9 prompts, used only in continuous analyses;
- higher-FOL reported group: 11 prompts.

Thus every formally reported source-level ASR group uses a prime denominator:
31 for the full source and 11 for each low/high FOL band. The middle band is not
given a standalone ASR. Prime denominators do not make percentages terminate at
two decimals, so all tables retain the integer numerator and denominator beside
the rounded percentage.

This prime-denominator rule applies only to the focused FOL validation. It does
not change the 50-example manifests used by the main fair-comparison matrix.

High and low bands are matched as closely as possible on source, risk category,
initial safety label, attack loss, prompt length, and perplexity. Matching
calipers and unmatched records are fixed before response generation and saved
in the manifest. Continuous analyses use all 62 prompts and do not depend on
the high/low grouping.

Five held-out prompts from each source are used to calibrate perturbation radii.
They are not part of the 31-prompt source validation sample and are not used in
hypothesis tests.

### Experiment A: Local Behavior Flips

Perturb only the soft prefix and editable seed-intent span. For each prompt:

1. evaluate radii `0.25 * epsilon`, `0.5 * epsilon`, `epsilon`, and
   `2 * epsilon`;
2. draw eight isotropic random directions per radius without selecting them by
   FOL, attack gradient, response, or judge score;
3. materialize each state and apply the common semantic-intent filter;
4. resample rejected directions up to 64 attempts per prompt and radius;
5. generate Qwen2.5-14B responses for all accepted candidates;
6. assign safety labels independently with the primary and secondary judges.

A judge-specific flip is a label change relative to the original prompt under
the same judge. The primary outcomes are:

```text
BFR(x, r) = accepted perturbations that flip / accepted perturbations at r
d50(x)    = radius at which an isotonic BFR-versus-radius fit reaches 0.5
```

If `d50` is not reached by `2 * epsilon`, it is treated as right-censored, not
imputed. A prompt-radius with fewer than four accepted directions is retained as
an acceptance failure but excluded from the corresponding rate estimate; the
failure rate is reported by FOL band to detect filtering bias.

### Experiment B: Internal Margin Boundary

For embedding state `e`, define:

```text
m(e) = mean normalized log p(answer anchors | e)
     - mean normalized log p(refusal anchors | e)
```

Fit a pre-registered monotonic calibration from `m(e)` to the primary judge's
unsafe probability using controlled-manifest examples outside the 62-prompt
validation sample. Define `m_star` as the margin where calibrated unsafe
probability is 0.5.

Along every accepted perturbation direction, locate the first crossing of
`m_star` using the fixed radius grid followed by binary search within the
crossing interval. This produces `d_margin`. Report its association with FOL
and its agreement with the observed behavior distance. WildGuard supplies a
robustness calibration; it does not replace the pre-registered primary margin
threshold.

### Experiment C: Safe-Unsafe Interpolation

For each validation prompt with semantically valid states on both sides of the
behavior boundary, choose the nearest opposite-label pair in editable-embedding
distance. Interpolate at 21 fixed positions:

```text
e(t) = (1 - t) * e_safe + t * e_unsafe,  t in {0.00, 0.05, ..., 1.00}
```

At each position record behavior label, internal margin, FOL, curvature, local
roughness, and semantic-filter outcome. A path is valid only when both endpoints
and at least 17 of 21 materialized positions preserve the seed intent.

For each valid path, measure:

- distance from the FOL maximum to the behavior crossing;
- distance from the FOL maximum to the `m_star` crossing;
- distance from the curvature maximum to each crossing;
- FOL concentration inside versus outside the crossing neighborhood.

If fewer than 13 valid interpolation paths are available, Experiment C is
reported as underpowered descriptive evidence and cannot satisfy H3.

### Alternative-Explanation Controls

For all 62 validation prompts, record:

- attack loss / ZOL;
- absolute calibrated margin distance `abs(m - m_star)`;
- prompt length and perplexity;
- local curvature estimated with four fixed-seed Hessian-vector products;
- local roughness estimated as attack-loss variance over eight micro-noise
  perturbations at `0.1 * epsilon`;
- semantic-filter acceptance rate.

These controls distinguish boundary proximity from high loss, steep curvature,
rough optimization geometry, and materialization artifacts.

### Statistical Analysis

The prompt, not each perturbation, is the independent sampling unit.

1. Estimate BFR-versus-radius curves by FOL band with prompt-level stratified
   bootstrap 95% confidence intervals.
2. Test H1 with a prompt-clustered logistic model containing radius, source,
   initial label, attack loss, and FOL, then add curvature and roughness in
   separately pre-specified sensitivity models to avoid overfitting.
3. Test H2 using rank association and an interval/right-censored boundary
   distance model, with confidence intervals bootstrapped by prompt.
4. Test H3 with a within-path permutation test comparing FOL-peak distance to
   random peak locations and curvature-peak distance.
5. Test H4 using grouped cross-validation by prompt. Compare a controls-only
   flip classifier with the same classifier plus FOL and report delta AUROC,
   delta AUPRC, calibration error, and bootstrap confidence intervals.
6. Apply Holm correction within the H1-H4 hypothesis family. Report effect
   sizes and confidence intervals in addition to corrected p-values.

Primary conclusions use Octopus-SEval labels. WildGuard must show the same
direction of effect for a robustness claim, even when its confidence interval
is wider.

### Claim Decision Rules

Use the following claim ladder without post-hoc relaxation:

- **Boundary-proxy support**: H1 and H2 are supported under the primary judge,
  effects have the same direction under WildGuard, and H3 shows FOL peaks closer
  to crossings than random locations. H4 provides additional supporting, not
  mandatory, evidence.
- **Local-sensitivity support only**: H1 is supported but H2 or H3 is not. The
  manuscript must describe FOL as locating locally safety-sensitive regions,
  not boundary-adjacent states.
- **No boundary support**: H1 is not supported or effects reverse across judges.
  Remove the boundary-proximity interpretation and report FOL only as an
  optimization signal.
- **Inconclusive**: confidence intervals are too wide, semantic acceptance is
  systematically FOL-dependent, or fewer than 13 interpolation paths are valid.
  Report the limitation without promoting a weaker result to confirmation.

## Statistical Analysis for the Main Matrix

New methods share the same 50-example source manifests, enabling paired
comparisons:

- stratified prompt-level bootstrap 95% confidence intervals for ASR and RCR;
- paired ASR differences against Init and ZOL-only;
- McNemar tests for paired binary outcomes;
- paired rank tests for continuous judge scores and semantic similarity;
- Holm correction within each pre-declared comparison family;
- method ranks repeated under both judges and all judge thresholds.

Bootstrap intervals estimate sample uncertainty. A single optimization seed
does not estimate optimizer-seed variance, and the paper must state this
limitation explicitly. Frozen PDF aggregates are excluded from these tests.

## Output Artifacts

All generated artifacts are stored under:

`outputs/results/reviewer_additions/`

Required layout:

```text
frozen_results.json
manifests/
  controlled_<dataset>.jsonl
  fol_boundary_jailbound.jsonl
  fol_boundary_s_eval.jsonl
optimization/<dataset>/<method>/
responses/<target_model>/<dataset>/<method>/
judgments/<judge>/<target_model>/<dataset>/<method>/
fol_boundary/
  perturbations.jsonl
  margin_crossings.jsonl
  interpolation_paths.jsonl
  controls.jsonl
analysis/
  main_metrics.csv
  paired_tests.csv
  judge_sensitivity.csv
  materialization_fidelity.csv
  fol_boundary_metrics.csv
figures/
tables/
run_manifest.json
```

JSONL records include schema version, run id, git working-tree revision,
configuration hash, model revisions, sample id, method, checkpoint, random seed,
status, and failure reason. Result aggregation reads records through structured
parsers and never scrapes logs.

## Figures and Tables

The revision should add:

1. a fair optimization table over seven sources on Qwen2.5-14B;
2. a compact transfer table over the four core target models;
3. an optimization-step curve at 0/25/50/100 updates;
4. a materialization-fidelity table with success and failure examples;
5. an S-Eval versus JailBound taxonomy and generation-process comparison;
6. a four-panel FOL diagnostic figure:
   - BFR versus radius by FOL band;
   - boundary distance versus FOL;
   - centered interpolation curves for FOL, margin, and curvature;
   - controls-only versus controls-plus-FOL predictive performance;
7. a judge and threshold sensitivity table.

Every table separates `frozen_pdf` and `new_run` visually and in its caption.

## Failure Handling and Resumability

- Write one atomic record per completed example and checkpoint.
- Resume by stable cell id and configuration hash.
- Never silently retry with changed hyperparameters.
- Record OOM, tokenizer, generation, materialization, semantic-filter, and judge
  failures separately.
- On OOM, reduce batch size or enable activation checkpointing without changing
  per-example budgets. Any other compatibility change creates a new config hash.
- A method remains in the denominator when its algorithm fails on a valid
  example; report both intention-to-evaluate and successful-execution summaries.
- Frozen results are read-only registry entries and are never overwritten by a
  resumed run.

## Verification Requirements

Before full execution, verify:

- dataset adapters produce stable ids and expected source counts;
- manifests contain no duplicate prompt hashes within a source;
- all methods receive identical ordered sample ids;
- cell identities skip only exact frozen/new matches;
- optimization budgets and checkpoints are enforced mechanically;
- semantic filtering is deterministic for a fixed encoder revision;
- ASR aggregation preserves numerator and denominator and rounds consistently;
- prompt-level bootstrap never resamples perturbations as independent prompts;
- FOL directions are not used to select random perturbation directions;
- margin calibration excludes the 62 validation prompts;
- a five-example smoke run completes optimization, materialization, generation,
  both judges, resume, and aggregation.

## Reviewer-Coverage Summary

This design fully addresses the empirical requests concerning fair baseline
optimization, PEZ/GBDA/GCG, no-FOL/ZOL-only, S-Eval, materialization fidelity,
threshold sensitivity, and uncertainty reporting. It substantially addresses
judge dependence and boundary interpretation but does not substitute for formal
human evaluation or a mathematical theorem.

The manuscript must separately revise:

- positioning relative to SAM/sharpness and automated jailbreak methods;
- the strength of the safety-boundary wording;
- limitations concerning materialization and the boundary assumption;
- dense tables, figure readability, and symbol definitions.
