# M3 Plan — Encoding Audit and Controlled Emotion Shifts

Status: **READY TO IMPLEMENT / RUN; NOT YET STARTED**

This is the next scientific milestone after the validated RAVDESS manifest/splits
(M1) and frozen WavLM extraction (M2).

Paper terminology update from the 2026-09-25 advisor meeting:

- use **confounder** for speaker identity, statement identity, and other labeled
  variation that is not the target emotion but may affect the classifier;
- keep the three questions separate:
  1. **Encoded** — can a classifier predict the confounder label?
  2. **Used** — after emotion training, does targeted test-time removal change
     the same classifier?
  3. **Harmful under shift** — if the confounder is suppressed during training
     and the emotion classifier is retrained, does OOD generalization improve?

M3 addresses the first question and establishes the controlled-shift baseline.
Do **not** start M4 intervention work until M3 is complete and reviewed.

## Inputs

Use the existing validated RAVDESS artifacts.

- Manifest: `artifacts/manifests/ravdess.parquet`
- Splits: `artifacts/manifests/ravdess_splits.parquet`
- WavLM cache:
  `artifacts/embeddings/ravdess/microsoft_wavlm-base-plus/d99979b610d68bef`
- Utterances: 1,440
- Layers: 13
- Hidden size: 768

Do not recompute WavLM embeddings.

## M3A — Layer-wise encoding audit

### Question

What emotion and confounder labels are linearly decodable from each WavLM layer?

### Targets

Run the same probe family for:

- emotion;
- speaker identity;
- statement identity;
- gender;
- emotion intensity.

Primary confounders for the paper are **statement identity** and
**speaker identity**. Gender and intensity remain secondary diagnostics.

### Split

Use the validated attribute-audit split:

- train: repetition 1;
- test: repetition 2.

This keeps the speaker and statement label spaces represented on both sides.

### Probe protocol

- regularized linear logistic regression;
- standardize features using training data only;
- use one small, predeclared regularization grid consistently across all
  layers/targets;
- perform model selection using training data only;
- no aggressive per-layer hyperparameter tuning;
- fix and record all random seeds.

Primary metric:

- macro-F1.

Also record:

- accuracy;
- appropriate majority/chance baseline where meaningful;
- selected regularization value;
- train/test counts.

### Required artifact

Produce one machine-readable table with at least:

```
model
dataset
layer
target
train_split
test_split
macro_f1
accuracy
baseline
selected_regularization
seed
```

Also produce:

- layer × target heatmap;
- compact JSON summary;
- exact config/command;
- tests for split alignment, feature/ID alignment, and finite values.

## M3B — Controlled emotion-generalization baseline

### Question

Where does emotion recognition degrade when the corresponding confounder changes?

For every WavLM layer, train the same class of linear emotion head.

### Evaluation conditions

1. **Near-IID reference**
   - train repetition 1 → test repetition 2.

2. **Lexical shift**
   - statement 1 → statement 2;
   - statement 2 → statement 1.

3. **Speaker shift**
   - reuse the deterministic held-out-speaker folds produced in M1;
   - never allow one speaker to appear in both train and test within a fold.

Gender shift remains secondary and is not required for the first M3 gate.

### Required metrics

For each layer/condition:

- macro-F1;
- accuracy;
- train/test counts;
- fold ID where applicable.

Compute:

`shift_gap = repetition_reference_macro_f1 - shifted_macro_f1`

For speaker shift, preserve the per-fold distribution and also report an
aggregate with uncertainty; do not keep only one summary number.

### Required artifacts

- machine-readable layer × shift table;
- emotion macro-F1 by layer and shift plot;
- lexical-shift summary in both directions;
- speaker-fold summary;
- compact JSON report.

## M3 scientific gate

Review M3 before implementing targeted removal.

Proceed to M4 only if all are true:

1. emotion is meaningfully decodable;
2. at least one primary confounder (statement or speaker) is clearly decodable;
3. at least one corresponding controlled shift produces a meaningful emotion
   generalization gap;
4. the result survives manifest/split/ID alignment sanity checks.

If this gate fails, diagnose the representation, pooling, labels, or split
semantics before adding models or datasets.

## Compute guidance

M3 uses cached 768-dimensional representations. It should be relatively cheap
and is primarily CPU / small-matrix work; a large GPU is not required.

Microsoft-approved hackathon compute may be used, but do not spend expensive GPU
time merely because it is available. Larger compute is more likely to matter for
later replication or intervention sweeps.

## Handoff requirements

At the end of each substantial run:

1. update `RUNS.md`;
2. update `STATUS.md` only when the milestone state changes;
3. commit code/config/artifact metadata;
4. push the branch;
5. transfer only reviewed, traceable scientific results to the paper repo's
   `RESULTS_LEDGER.md`.
